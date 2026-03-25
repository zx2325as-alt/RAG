# 智能知识库与模型微调平台 (RAG & Fine-tuning Platform) 技术白皮书

## 1. 系统概述

本项目是一个企业级的人工智能平台，深度集成了 **RAG（检索增强生成）** 架构与 **LLM 微调（Fine-tuning）** 模块。平台旨在解决企业内部垂直领域（如 IT 运维、产品知识库）的信息孤岛问题，通过本地化知识库构建与私有化大模型微调，实现高精度、低延迟、数据绝对安全的智能问答。

本白皮书面向开发人员、架构师和算法工程师，详细阐述系统的技术选型、架构设计、核心工作流以及代码层面的实现细节。

***

## 2. 技术栈与生态选型

### 2.1 后端服务与核心框架

- **Web 框架**: `Flask`。选择 Flask 是因为其轻量、灵活，非常适合构建提供 SSE (Server-Sent Events) 流式输出的 AI 中台 API。
- **数据库 ORM**: `SQLAlchemy`。向下兼容 SQLite (适合快速验证与本地轻量级部署) 与 MySQL (适合生产环境)。
- **异步与并发**: 采用 Python 原生 `threading` 模块结合 `Flask App Context` 实现文档的后台异步解析与向量化，避免阻塞主 Web 线程。

### 2.2 RAG (检索增强) 相关技术

- **大语言模型编排**: `LangChain`。用于构建 Prompt 模板、封装 LLM 调用以及管理对话历史 (Memory)。
- **文档解析与预处理**:
  - `PyMuPDF` (`fitz`): 高效提取 PDF 文本层。
  - `pdfplumber`: 精准提取 PDF 中的表格，并将其格式化为 Markdown 结构，避免数据揉碎。
  - `python-docx`: 解析 Word 文档，同样支持表格的 Markdown 转换。
  - `PaddleOCR`: 针对无文本层的扫描版 PDF 或图片，提供本地化的高精度 OCR 识别（带有页数限制保护策略）。
- **文本切分 (Chunking)**: `RecursiveCharacterTextSplitter`。
- **向量检索 (Dense Retrieval)**: `FAISS` (Facebook AI Similarity Search)。将文本向量化后存储于本地磁盘，内存占用极低且查询速度达毫秒级。
- **稀疏检索 (Sparse Retrieval)**: `rank_bm25`。传统的基于词频的检索算法，用于弥补向量模型在“专有名词”、“错误代码”上的精确匹配缺陷。

### 2.3 基础 AI 模型库 (Model Hub)

- **Embedding 模型**: `BAAI/bge-large-zh`。负责将文本切片转化为高维稠密向量。
- **Reranker 模型**: `BAAI/bge-reranker-large`。交叉编码器，用于对 FAISS 和 BM25 召回的混合结果进行二次打分重排，极大提升 Top-K 的相关性。
- **LLM 推理引擎**: 支持通过 OpenAI 兼容协议接入 `Ollama` 和 `vLLM`（本地私有化部署），以及 `DeepSeek`、`通义千问`等公有云 API。

### 2.4 模型微调框架 (Fine-tuning)

- **底层引擎**: `LLaMA-Factory`。当前业界最流行的低代码/无代码微调框架。
- **集成方式**: 通过 Python `subprocess.Popen` 在后台拉起独立的 `llamafactory-cli train` 进程，与主 Web 服务解耦，防止 OOM 导致 Web 崩溃。
- **监控**: 集成 `TensorBoard` 进行训练 Loss 的实时可视化。

***

## 3. 系统架构设计

系统采用**前后端分离**、**读写分离**以及**微服务化解耦**的设计理念。

```JSON
graph TD
    subgraph Frontend [前端展示层 (HTML/JS/Bootstrap)]
        UI_QA[智能问答对话 UI]
        UI_KB[知识库管理面板]
        UI_FT[微调参数配置与看板]
    end

    subgraph APIGateway [API 路由网关层 (routes.py)]
        API_Query[POST /query<br>SSE 流式问答]
        API_Upload[POST /upload<br>异步文档解析]
        API_FT[POST /finetune<br>任务下发]
    end

    subgraph CoreServices [核心业务逻辑层 (Service Layer)]
        DocService[DocumentService<br>多模态解析/表格提取/分块]
        KBService[KnowledgeBaseService<br>多路召回/RRF融合/重排]
        QAService[QAService<br>Prompt工程/LLM调用/内联引用]
        FTService[FinetuneService<br>YAML生成/进程拉起]
    end

    subgraph DataStorage [数据持久化层]
        DB[(关系型数据库)<br>文档元数据/Chunk映射]
        VS[(FAISS 向量库)<br>db_name/index.faiss]
        BM25[(BM25 索引)<br>内存驻留]
    end

    subgraph AI_Engine [AI 推理与训练底座]
        Embed[bge-large-zh]
        Rerank[bge-reranker-large]
        LLM[vLLM / Ollama / API]
        LlamaFactory[LLaMA-Factory 训练进程]
    end

    Frontend <--> APIGateway
    APIGateway --> CoreServices
    DocService --> DB
    DocService --> VS
    DocService --> BM25
    KBService <--> VS
    KBService <--> BM25
    KBService <--> Embed
    KBService <--> Rerank
    QAService <--> LLM
    FTService --> LlamaFactory
```

***

## 4. 核心工作流详解 (Workflows)

### 4.1 文档导入与知识库构建流 (Data Ingestion Pipeline)

为了保证大文件上传时系统的鲁棒性和响应速度，我们采用了**异步非阻塞+批量写入**的策略：

1. **上传与持久化**: 用户通过前端上传 PDF/Word 文件。API 网关接收后立即保存至本地 `uploads/` 目录，并在数据库 `Document` 表中创建状态为 `pending` 的记录，随后**立即返回 HTTP 200** 给前端。
2. **异步触发**: API 层开启 `threading.Thread` 后台线程调用 `DocumentService.parse_document()`。
3. **多模态提取**:
   - 检测到 PDF，优先使用 `pdfplumber` 提取表格并转为 Markdown (`| 表头 |`)。
   - 使用 `PyMuPDF` 提取普通文本。若文本量极少（如截图），则启用可控的 `PaddleOCR` 识别（限制最大页数以防止 GPU/CPU 阻塞）。
4. **语义切分**: 结合 LangChain 的 `RecursiveCharacterTextSplitter`，按照固定长度（伴随 Overlap）进行切分。
5. **批量落库**: 使用 SQLAlchemy 的 `bulk_save_objects` 将上万个 Chunk 毫秒级写入数据库，获取 `chunk_id`。
6. **增量向量化**: 将 Chunk 送入 BGE Embedding 模型，增量追加至 FAISS 索引并持久化到磁盘；同时更新内存中的 BM25 词频索引。

### 4.2 混合检索与增强生成流 (Hybrid Search & RAG Pipeline)

当用户在对话框输入问题（如：“Zabbix 报错 500 如何排查？”）时：

1. **意图改写 (Query Rewrite)**: 将用户的简短输入结合历史上下文，利用 LLM 改写为完整的独立查询语句。
2. **双路召回 (Hybrid Retrieval)**:
   - **路 A (Dense)**: 将 Query 向量化，去 FAISS 库中计算 L2 距离，召回 Top-K（如 10 个）Chunk。
   - **路 B (Sparse)**: 对 Query 进行 Tokenize，去 BM25 库中计算 TF-IDF 变种得分，召回 Top-K Chunk。
3. **RRF 排序融合 (Reciprocal Rank Fusion)**: 将两路召回的结果不看绝对分值，仅依据排名进行 RRF 公式融合，消除异构得分差异，取 Top-50。
4. **交叉重排与父子联想 (Rerank & Small-to-Big)**:
   - 将 Query 与 Top-50 的 Chunk 拼接为 Pair，送入 `bge-reranker` 模型进行高精度打分，截取最终的 Top-5。
   - **关键优化**：针对这 5 个核心切片，根据其 `doc_id` 和 `chunk_index` 向数据库**回溯查询其前驱和后继切片**，将其扩展为一个完整的长段落，提供完美的上下文。
5. **Prompt 构建与内联引用**: 将扩展后的文本拼入 Prompt，并明确标注 `资料 [1]`。要求 LLM 输出带有 `[1]` 角标的规范回答。
6. **SSE 流式返回**: 通过 `stream_with_context` 逐字将大模型生成的 Token 推送给前端进行打字机渲染。

### 4.3 强化学习反馈与微调流 (RLHF / SFT Pipeline)

1. **用户反馈收集**: 问答结束后，用户对生成的答案进行评分（1-5星）。
2. **数据集自动沉淀**: 若评分低于 4 分且用户提供了纠正建议，系统自动将 `(Question, Bad Answer, Correct Answer)` 构建为 DPO（直接偏好优化）或 SFT（指令微调）的 JSONL 格式数据。
3. **动态重分析**: 若评分极低，系统会自动扩大检索范围（Top-15），注入更强硬的 System Prompt，要求大模型推翻之前的结论进行深度二次排障分析。
4. **一键微调拉起**: 运维人员积累足够数据后，在后台看板设置 Epoch、Learning Rate，点击训练。系统将参数组装为 YAML，拉起 `LLaMA-Factory` 进程，并通过 SSE 读取 `stdout` 实时展示训练进度与 Loss 曲线。

***

## 5. 工程实践与性能优化亮点

1. **解决 ORM 脱机错误 (DetachedInstanceError)**:
   在 SSE 流式生成器中，由于脱离了原本的 Flask Request Context，访问惰性加载（Lazy Load）的关联表属性时会崩溃。我们在 `KnowledgeBaseService` 中采用了主动拉起 `app_context()` 并按需发起 `db.session.query()` 的防断连机制，极大提升了流式接口的健壮性。
2. **零延迟的内联引用 (Inline Citation)**:
   传统做法通过大模型抽取实体再做二次匹配来生成引用，这会导致严重的延迟。我们通过纯 Prompt 工程结合前端正则表达式（`<span class="badge">`），在不增加任何额外推理耗时的前提下，实现了完美的来源溯源高亮。
3. **环境动态切换隔离**:
   通过单一的 `.env` 文件配合 `app/config.py` 中的类继承策略（`LocalConfig` vs `ProductionConfig`），实现了单机开发（绑定 `127.0.0.1`）与线上部署（绑定 `0.0.0.0`）的网络安全隔离。

***

## 6. 目录结构说明

```text
RAG/
├── app/
│   ├── api/
│   │   └── routes.py           # 核心 API 路由层 (负责请求转发、异步任务派发、SSE 流式响应)
│   ├── db/
│   │   ├── models.py           # SQLAlchemy 数据表模型 (Document, Chunk, QueryLog 等)
│   │   └── __init__.py         # 数据库初始化
│   ├── services/
│   │   ├── document_service.py # 文档解析、预处理、入库服务
│   │   ├── knowledge_base_service.py # FAISS 向量库与 BM25 混合检索、RRF 重排服务
│   │   └── qa_service.py       # 大模型 Prompt 构建与生成服务
│   ├── static/
│   │   ├── css/style.css       # 前端样式文件
│   │   ├── js/main.js          # 前端核心交互逻辑 (SSE 解析、引用高亮等)
│   │   └── vendor/             # 本地化第三方库 (Bootstrap, jQuery, Chart.js)
│   ├── templates/
│   │   ├── index.html          # 知识问答与上传管理主界面
│   │   ├── finetune.html       # 模型微调配置与控制台面板
│   │   └── eval_report.html    # 训练评估与 Loss 曲线展示界面
│   ├── utils/
│   │   ├── file_processor.py   # 多模态文件提取器 (PyMuPDF, pdfplumber, PaddleOCR)
│   │   └── text_processor.py   # 文本切分器 (RecursiveCharacterTextSplitter)
│   ├── config.py               # 多环境配置管理 (LocalConfig, ProductionConfig)
│   └── main.py                 # Flask 应用启动入口
├── data/                       # 存放微调生成的 JSONL 数据集
├── finetune_configs/           # 自动生成的 LLaMA-Factory 训练 YAML 配置文件
├── hugface/                    # 本地 Hugging Face 模型缓存目录
├── model/                      # 本地 Embedding 与 Rerank 模型权重存放路径
├── faiss_index/                # FAISS 向量索引本地持久化存储目录
├── logs/                       # 系统运行日志与 TensorBoard 日志
├── .env                        # 环境变量文件 (配置 APP_ENV 等)
├── requirements.txt            # Python 依赖清单
└── README.md                   # 本技术白皮书
```

***

## 7. 部署与快速启动指南

### 7.1 环境准备

1. **操作系统**: 推荐 Ubuntu 22.04 / CentOS 7+ / Windows 10+ (需支持 WSL2 或直接 Python 环境)。
2. **硬件要求**:
   - **CPU/内存**: 至少 8核 16GB RAM（仅运行 RAG 检索）。
   - **GPU (可选但推荐)**: NVIDIA GPU，显存 >= 8GB（用于加速 BGE Embedding/Rerank 推理），若需本地微调 7B 模型，推荐 24GB+ 显存。
3. **软件依赖**: Python 3.10+，CUDA 11.8+ (若使用 GPU)。

### 7.2 安装步骤

```bash
# 1. 克隆代码仓库
git clone <repository_url>
cd RAG

# 2. 创建并激活虚拟环境 (推荐使用 Conda)
conda create -n rag_env python=3.10 -y
conda activate rag_env

# 3. 安装核心依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. (可选) 安装 LLaMA-Factory (如需进行模型微调)
pip install llamafactory -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. (可选) 安装 PaddleOCR 及其依赖 (如需处理纯图片或扫描版PDF)
pip install paddlepaddle-gpu paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 7.3 模型下载与放置

请从 Hugging Face 或 ModelScope 下载以下两个核心基础模型，并放置于项目根目录的 `model/` 文件夹下：

1. `bge-large-zh` (Embedding模型) -> `RAG/model/bge-large-zh/`
2. `bge-reranker-large` (重排模型) -> `RAG/model/bge-reranker-large/`

*(注：系统* *`config.py`* *默认读取此相对路径，若路径不同，请在* *`.env`* *中覆盖配置)*

### 7.4 配置文件设置

在项目根目录创建或修改 `.env` 文件：

```ini
# 运行环境：local 或 production (决定了服务绑定 127.0.0.1 还是 0.0.0.0)
APP_ENV=local

# 默认在线大模型配置
ACTIVE_LLM=deepseek
DEEPSEEK_API_KEY=sk-your-api-key-here

# 如果使用本地 vLLM 或 Ollama
# VLLM_API_URL=http://127.0.0.1:8000/v1
# OLLAMA_BASE_URL=http://127.0.0.1:11434
```

### 7.5 启动服务

```bash
# 确保在 RAG 项目根目录下执行
python app/main.py
```

服务启动后，默认可通过浏览器访问：`http://127.0.0.1:6008`。

***

## 8. 核心 API 接口文档 (供二次开发参考)

平台前后端通过标准的 RESTful API 和 SSE 进行通信，以下为部分核心接口说明：

### 8.1 文档上传与知识库构建

- **URL**: `/upload`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **参数**:
  - `files[]`: (File) 需要上传的文档集合（支持多选）。
  - `db_name`: (String, Optional) 目标知识库名称，默认为 `default`。
- **返回值**: 立即返回 `200 OK`，包含 `{"message": "All files uploaded successfully"}`。实际的解析、向量化过程通过 Python 后台线程异步执行，不阻塞当前请求。

### 8.2 知识检索与流式生成

- **URL**: `/query`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **参数**:
  - `question`: (String) 用户提问。
  - `db_names`: (Array\[String]) 需要检索的知识库范围。
- **返回值**: `text/event-stream` (SSE 流)。
  - `{"type": "chunk", "content": "..."}`: 增量生成的回答 Token。
  - `{"type": "sources", "sources": [{"doc_name": "...", "content": "...", "score": 0.98}]}`: 最终返回的被采纳的参考文献元数据。

### 8.3 触发模型微调

- **URL**: `/api/finetune/start`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **参数**: 包含 `dataset`, `model_name`, `learning_rate`, `batch_size`, `epochs` 等超参数。
- **返回值**: `text/event-stream`。持续推送 `llamafactory-cli` 的控制台标准输出，直至训练结束。

***

## 9. 常见问题排查 (FAQ)

**Q1: 为什么上传大型 PDF 文件时速度极慢，甚至导致服务器内存/显存耗尽？**
**A**:

- 检查该 PDF 是否为全扫描件（无文本层）。若是，系统会自动回退使用 `PaddleOCR` 逐页识别图片，这是一个非常耗时的深度学习推理过程。
- **解决方案**: 我们已在代码 (`file_processor.py`) 中限制了单文档的最大 OCR 页数 (`max_ocr_pages = 5`)。若仍卡顿，可尝试在部署环境中彻底卸载 `paddleocr`，系统将优雅降级，跳过图片识别。

**Q2: 提问时终端报错** **`sqlalchemy.orm.exc.DetachedInstanceError`** **导致回答中断。**
**A**:

- **原因**: 发生在 SSE 流式响应阶段。因为流式生成器在 `yield` 时，原本的 Flask Request Context 可能已结束，数据库 Session 被回收。此时访问惰性加载的属性（如 `chunk.document.doc_name`）会引发此异常。
- **解决方案**: 该问题已在最新版本中修复。我们在 `KnowledgeBaseService` 的检索方法中捕获了该异常，并使用 `with current_app.app_context():` 重新发起短连接查询，确保流式输出的绝对稳定。

**Q3: 为什么有时候大模型回答很好，但底部的“参考资料”却显示“抱歉，知识库中没有相关信息”？**
**A**:

- **原因**: 用户的提问触发了代码内置的 **Agentic 路由拦截**（例如：包含“告警次数”或“数据库查询”等字眼）。此时系统模拟了 Text2SQL 插件行为直接返回了设定好的数据库结果，跳过了 FAISS 检索。
- **调整**: 若需修改或禁用此类硬编码的 Agent 拦截，请查阅 `qa_service.py` 中的 `Agent 路由` 相关代码段。

**Q4: 微调启动失败，提示** **`DatasetGenerationError: Please pass features or at least one example`。**
**A**:

- **原因**: 提交的微调数据集为空，或者 JSONL 格式不合法导致 LLaMA-Factory 无法加载。
- **解决方案**: 我们在 `/api/finetune/start` 接口中加入了严格的数据集落盘校验，如果体积为 0 将直接在前端报错拦截。请确保在网页端配置数据集时，至少包含一条有效的 Instruction/Output 记录。

***

## 10. 后续演进规划 (Roadmap)

1. **Graph RAG 集成**: 引入图数据库（如 Neo4j），利用 LLM 在离线阶段抽取服务器、IP、业务系统之间的拓扑关系。在检索阶段结合向量检索与图谱路径遍历，解决“牵一发而动全身”的复杂排障问题。
2. **Agentic RAG (智能体化)**: 增加更多的 Function Calling 插件。除了现有的 Text2SQL 拦截，未来接入 Zabbix API、K8s API，让系统不仅能“查文档”，还能直接“查实时机器状态”。
3. **分布式向量库迁移**: 当知识库切片达到千万级别时，考虑将本地 FAISS 内存实例无缝迁移至 Milvus 或 Qdrant 分布式向量数据库。

