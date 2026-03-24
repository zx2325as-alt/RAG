# 智能知识库与模型微调系统技术文档

## 1. 项目简介

本项目是一个集成了 **RAG（Retrieval-Augmented Generation，检索增强生成）** 与 **LLM 微调（Fine-tuning）** 的综合性 AI 平台。系统支持多种格式文档的上传与解析，通过构建本地向量知识库实现高精度的智能问答。同时，平台集成了模型微调看板，支持对本地（Ollama、vLLM）及在线开源大模型进行参数化微调，以满足特定垂直领域的业务需求。

---

## 2. 核心功能点及实现方式

### 2.1 智能知识库构建与管理 (RAG)
- **多格式文档支持**：支持 PDF、Word、TXT、Markdown 等多种格式文档的上传。
  - *实现方式*：前端通过 FormData 异步上传文件，后端基于 `LangChain` 的 Document Loaders（如 `PyMuPDFLoader`, `Docx2txtLoader`, `TextLoader`）进行解析。
- **文档分块与向量化**：将长文档切分为语义连贯的 Chunk，并转换为高维向量。
  - *实现方式*：使用 `RecursiveCharacterTextSplitter` 进行文本分块；采用本地部署的 `bge-large-zh` 模型作为 Embedding 模型将文本转化为向量。
- **多库管理与检索控制**：支持创建多个知识库（如默认库、网络库、设备库等），查询时可多选范围。
  - *实现方式*：后端通过 `FAISS` 维护多个独立的向量索引实例，并将文本块（Chunks）持久化存储在 SQLite/MySQL 数据库中。

### 2.2 智能问答与混合检索
- **流式问答对话**：提供类似 ChatGPT 的打字机效果问答体验。
  - *实现方式*：前端使用 `EventSource` (SSE) 接收服务器的流式事件 (`text/event-stream`)；后端调用大模型的 `stream` 接口并使用 Flask 的 `stream_with_context` 实时推送数据。
- **双路召回与重排序 (Hybrid Search + Rerank)**：提高检索准确率，减少大模型幻觉。
  - *实现方式*：
    1. **向量检索 (Dense Retrieval)**：使用 FAISS 结合 Embedding 模型进行语义相似度检索。
    2. **关键词检索 (Sparse Retrieval)**：基于 `BM25` 算法进行精准字词匹配。
    3. **重排序 (Reranking)**：利用 `bge-reranker-large` 模型对双路召回的结果进行交叉打分，重新排序并截断提取 Top-K，最后送入大模型生成最终答案。

### 2.3 大模型灵活切换
- **多模型引擎无缝对接**：支持在线 API 与本地私有化部署模型的动态切换。
  - *实现方式*：前端提供联动下拉框（分类 -> 具体模型）。后端在 `QAService` 中根据配置实例化不同的 `LangChain ChatOpenAI` 客户端。
  - **支持的引擎**：
    - **在线 API**：DeepSeek, ChatGPT, 通义千问等（基于 OpenAI 兼容格式）。
    - **Ollama**：管理和调用本地 GGUF 量化模型。
    - **vLLM**：本地高并发推理引擎。

### 2.4 可视化模型微调 (Fine-tuning)
- **数据集在线构建**：支持在网页端直接构建、校验并保存 JSONL 格式的微调数据集。
  - *实现方式*：前端提供动态表单添加对话指令（Instruction, Input, Output），后端验证后生成符合 `LLaMA-Factory` 规范的数据集并自动更新 `dataset_info.json`。
- **超参数配置与任务拉起**：提供图形化界面配置 LoRA 微调参数（如 Rank, Alpha, 学习率, Batch Size 等）。
  - *实现方式*：后端接收参数后生成 YAML 配置文件，通过 Python 的 `subprocess.Popen` 后台拉起 `llamafactory-cli train` 进程。
- **模型智能映射与兜底**：解决本地引擎模型格式与微调框架的兼容性问题。
  - *实现方式*：针对缺少 `config.json` 的 Ollama/vLLM 本地运行模型，系统内置了智能映射字典，自动回退拉取对应的 HuggingFace 官方未量化基座模型进行训练。
- **实时日志与监控**：
  - *实现方式*：通过 SSE 实时将底层微调命令的控制台输出（stdout/stderr）推送到前端；同时后台自动启动 `TensorBoard` 服务，供用户查看 Loss 和 F1 曲线。

### 2.5 环境一键切换配置
- **本地与线上配置分离**：
  - *实现方式*：基于面向对象的 `Config` 类设计（`BaseConfig`, `LocalConfig`, `ProductionConfig`），通过 `.env` 文件中的 `APP_ENV` 环境变量一键切换数据库连接、绑定 IP、模型绝对路径等核心参数，避免部署时的大量修改。

---

## 3. 技术栈选型

### 3.1 后端技术
- **核心框架**: `Flask` (轻量级，便于快速构建 API 与 SSE 流式响应)
- **AI 编排框架**: `LangChain` (用于串联文档加载、向量存储、提示词模板与大模型调用)
- **数据库 ORM**: `SQLAlchemy` (兼容 SQLite/MySQL，管理文档元数据与日志)
- **向量数据库**: `FAISS` (Facebook AI Similarity Search，本地高性能向量检索引擎)
- **文本匹配**: `rank_bm25` (用于稀疏检索)
- **微调框架**: `LLaMA-Factory` (业界主流的大模型低代码微调框架)

### 3.2 基础模型 (Base Models)
- **Embedding 向量模型**: `BAAI/bge-large-zh` (中文文本表征)
- **Rerank 重排模型**: `BAAI/bge-reranker-large` (检索结果二次打分)
- **LLM 引擎**: `Ollama`, `vLLM`

### 3.3 前端技术
- **UI 框架**: `Bootstrap 5` (响应式布局与现代化组件)
- **交互与 DOM 操作**: `jQuery` & 原生 JavaScript
- **异步通信**: `AJAX` (基础接口请求) & `Server-Sent Events (SSE)` (流式对话与实时日志推送)

---

## 4. 整体系统架构图

系统采用**前后端分离 + 核心微服务化**的架构设计：

```mermaid
graph TD
    %% 前端层
    subgraph Frontend [前端展示层 (HTML + JS + Bootstrap)]
        A1[知识库构建看板]
        A2[智能问答聊天界面]
        A3[模型微调与监控看板]
    end

    %% API 路由层
    subgraph API_Layer [Flask API 路由层]
        B1[/文档上传与解析 API/]
        B2[/流式查询与检索 API/]
        B3[/模型切换与配置 API/]
        B4[/微调任务调度 API/]
    end

    %% 核心业务逻辑层
    subgraph Core_Services [核心业务服务层]
        C1[DocumentService<br>文档分块与清洗]
        C2[KnowledgeBaseService<br>多路召回与混合检索]
        C3[QAService<br>Prompt构建与大模型调用]
        C4[FinetuneService<br>数据集管理与命令生成]
    end

    %% 数据与存储层
    subgraph Storage [数据与存储层]
        D1[(SQL Database)<br>SQLite/MySQL]
        D2[(FAISS Indexes)<br>本地向量索引池]
        D3[本地文件系统<br>数据集/配置/上传文件]
    end

    %% 模型与引擎层
    subgraph AI_Engines [底层 AI 模型与推理引擎]
        E1[bge-large-zh<br>Embedding]
        E2[bge-reranker-large<br>Rerank]
        E3[Ollama / vLLM<br>本地推理引擎]
        E4[LLaMA-Factory<br>底层微调进程]
        E5[在线大模型 API<br>DeepSeek/OpenAI]
    end

    %% 关联关系
    Frontend <-->|HTTP/AJAX & SSE| API_Layer
    
    API_Layer --> C1
    API_Layer --> C2
    API_Layer --> C3
    API_Layer --> C4

    C1 --> D1
    C1 --> D3
    C1 --> E1
    
    C2 --> D2
    C2 --> E1
    C2 --> E2
    
    C3 --> C2
    C3 --> E3
    C3 --> E5
    
    C4 --> D3
    C4 --> E4
```

## 5. 架构亮点说明
1. **服务单例模式**：在 Flask 的 `current_app` 上下文中维护 `QAService` 和 `KnowledgeBaseService` 的单例，避免了 FAISS 索引在内存中的重复加载和状态不一致问题。
2. **混合检索架构**：向量检索擅长“语义泛化”，BM25 擅长“专有名词精确匹配”。两者结合后通过交叉编码器 (Reranker) 重新打分，是目前企业级 RAG 系统的最佳实践方案。
3. **松耦合微调流**：微调模块通过 `subprocess` 与核心 Web 服务解耦。即便微调进程占用大量 GPU 资源或发生异常崩溃，Web 平台依然能保持稳定运行并提供日志追踪。