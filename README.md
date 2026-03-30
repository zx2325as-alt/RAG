# RAG 项目全流程深度解析与工程化实践指南

本项目是一个面向运维领域的智能问答与知识图谱系统，核心基于 RAG（检索增强生成）与 Agent（智能体）技术。本指南从**离线数据入库**、**在线检索与生成**、**工程化与 Agent 集成**三大阶段，详细梳理技术选型、参数配置、优化策略及问题解决方法。

---

## 第一阶段：离线数据入库 (Offline Data Ingestion)

离线数据入库是将企业的非结构化或半结构化文档（PDF、Word、Markdown 等）转化为可被机器检索的向量、关键词和知识图谱节点的关键过程。

### 核心代码执行流程
1. **API 接收文件**：用户通过调用 `app/api/routes.py` 中的 `upload_document()` 接口上传文件。
2. **生成文档记录**：调用 `app/services/document_service.py` 中的 `DocumentService.process_document(filepath, filename, db_name)`。
3. **文本解析 (Extraction)**：
   - 内部调用 `app/utils/file_processor.py` 中的 `extract_text_from_file(file_path)`。
   - 根据文件后缀路由：
     - PDF: 调用 `extract_text_from_pdf()` (结合 PyMuPDF 和 pdfplumber)。
     - Word: 调用 `extract_text_from_docx()`。
     - 图片: 调用 `extract_text_from_image()` (触发 RapidOCR GPU 加速提取)。
4. **文本清洗与分块 (Cleaning & Chunking)**：
   - 调用 `app/utils/text_processor.py` 中的 `clean_text(raw_text)` 去除无效空白符。
   - 调用 `detect_text_type_and_adjust_size()` 动态决定 Chunk 大小 (300/600/800)。
   - 调用 `chunk_text(cleaned_text, is_markdown, metadata)` 进行语义边界切分，并注入 `【文档元数据】`。
5. **入库持久化 (Ingestion)**：
   - 关系型数据：在 `process_document` 中，将切分好的 Chunk 批量保存至 MySQL (`db.session.bulk_save_objects(db_chunks)`)。
   - 向量与关键词：调用 `app/services/knowledge_base_service.py` 中的 `KnowledgeBaseService.add_texts(chunks, db_name)`。
     - 内部通过 Embedding 模型将文本转化为向量并加入 `self.vector_stores` (FAISS)。
     - 将分词后的 Token 加入 `self.bm25_stores` 建立 BM25 倒排索引。
   - 最后调用 `self.save_index(db_name)` 将 FAISS 与 BM25 持久化到本地磁盘。

### 1. 技术选型
*   **文档解析**：
    *   `PyMuPDF (fitz)`：负责基础 PDF 文本与内嵌图片的快速提取。
    *   `pdfplumber`：专门负责 PDF 中复杂表格的解析，并转化为 Markdown 格式。（注：已通过屏蔽 `pdfminer` 底层警告优化了日志输出体验）。
    *   `python-docx` / `openpyxl`：处理传统 Office 文档。
    *   `RapidOCR-onnxruntime`：针对扫描件、纯图片型 PDF 的 OCR 识别（已开启动态 GPU 探测，支持 CUDA 加速，保障识别效率与稳定性）。
*   **文本分块 (Chunking)**：基于 LangChain `RecursiveCharacterTextSplitter` 封装的智能动态分块器。
*   **向量化模型 (Embedding)**：`bge-large-zh`（支持 GPU 加速），开源且在中文场景表现优异。
*   **存储引擎**：
    *   **关系型数据**：MySQL (SQLAlchemy ORM)，存储文档元数据（MD5、上传状态）与分块明细。
    *   **向量检索库**：FAISS-CPU（通过 LangChain 集成），持久化在本地文件系统。
    *   **倒排索引库**：BM25 (`rank_bm25`)，用于稀疏检索（关键词精确匹配）。
    *   **图数据库**：Neo4j，存储从文档中抽取出的运维实体和关系，构建 Graph RAG。

### 2. 参数配置
*   **智能分块策略 (Dynamic Chunking)**：
    *   **动态大小**：不再使用固定 300 字符。系统会根据内容密度探测，若包含代码块 (```` ````) 或 JSON 则自动扩展至 800 字符；若日志密度高则扩展至 600 字符；常规文本保持 300 字符，重叠窗口 `chunk_overlap = 50`。
    *   **语义边界优先**：分隔符优先级严格遵循标点符号级别：`["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", ";", " ", ""]`，确保在句子边界切分，防止语义被硬截断。
*   **向量化配置**：
    *   启用 BGE 推荐的 `normalize_embeddings=True`（余弦相似度优化）。
    *   设备自动选择：`'cuda' if torch.cuda.is_available() else 'cpu'`。

### 3. 优化策略
*   **多模态解析增强**：
    *   如果文档是纯扫描件（每页文本数 < 50 且无明确图片提取），将整页通过 `fitz.Matrix(2, 2)` 提高分辨率渲染为图片，再输入 OCR 提取。
    *   将表格结构转换为 Markdown 格式 (`| --- |`)，最大限度保留表格的行列语义关系，方便大模型理解。
*   **语义与层级切分**：对 `.md` 文件，优先使用 `MarkdownHeaderTextSplitter` 按照 H1/H2/H3 层级进行切分。
*   **强元数据注入**：在分块入库时，不仅仅注入标题层级，还会将文档来源 (`source`)、文件类型 (`type`) 和入库时间 (`created_at`) 组合成 `【文档元数据】` 拼接到 Chunk 头部，增强后续检索的区分度。
*   **高并发数据库写入**：切分后的成百上千个文本块，采用 `db.session.bulk_save_objects(db_chunks)` 批量写入 MySQL，大幅降低 IO 耗时。

### 4. 问题解决方法
*   **中文文件名乱码与过滤问题**：
    *   *问题*：Werkzeug 原生的 `secure_filename` 会将纯中文文件名（如 `人工智能.pdf`）过滤为空，导致存储报错。
    *   *解决*：自定义过滤规则，仅替换目录穿越符（`.replace('/', '_').replace('\\', '_')`），保留中文字符。
*   **ORM N+1 查询与 DetachedInstanceError**：
    *   *问题*：在循环向 FAISS 注入数据时，频繁触发文档表单行查询，导致性能急剧下降或产生会话脱离报错。
    *   *解决*：使用 `doc_id.in_` 在循环外预先批量获取所有的 `doc_map`，在内存中完成组装。

---

## 第二阶段：在线检索与生成 (Online Retrieval and Generation)

此阶段是系统的核心大脑，负责理解用户意图、从海量库中召回高价值知识，并利用大语言模型（LLM）生成专业解答。

### 核心代码执行流程
1. **接收用户提问**：前端请求 `app/api/routes.py` 中的 `query()` 接口，内部触发 SSE 流式响应，调用 `app/services/qa_service.py` 中的 `QAService.stream_answer_question(question, user_id, db_names)`。
2. **多轮对话处理与问题重构**：
   - 提取 Redis 中的聊天历史 `self.get_session_history(user_id)`，进行动态裁剪（仅保留最近 4 条消息）。
   - 构建 `rewrite_prompt`，调用 `self.llm.invoke()` 让模型补全指代词，生成独立的 `search_query`，防止历史消息干扰检索质量。
3. **多路混合召回 (Hybrid Retrieval)**：
   - 调用 `app/services/knowledge_base_service.py` 中的 `KnowledgeBaseService.search(search_query)`。
   - **图谱检索**：调用 `self.graph_search()` 进行 1 到 2 跳的动态 Cypher 查询，并将结构化结果转化为自然语言 `doc`。
   - **向量检索**：调用 `self.vector_stores[db_name].similarity_search_with_score(query)` 召回 Top-30/60 稠密结果。
   - **稀疏检索**：调用 `self.bm25_stores[db_name].get_scores()` 召回 Top-30/60 关键词匹配结果。
4. **自适应融合与重排 (RRF & Rerank)**：
   - 在 `search()` 内部，基于意图探测（是否包含英文缩写）动态分配 BM25 和 Vector 的 RRF 融合权重。
   - 将融合后的 Top-100 结果送入 `bge-reranker-large`。
   - 通过 `self.reranker_model(**inputs).logits` 获取交叉注意力打分，倒序排列并结合之前命中的 `self.reranker_cache` 缓存，最终截取 Top-5 结果返回。
5. **质量兜底与生成拦截**：
   - 回到 `qa_service.py`，触发质量硬拦截：如果 `top_score < 0.6` 且结果数 `< 2`，则直接中断生成流，向前端返回：“抱歉，当前知识库中未找到相关资料...”。
6. **LLM 流式生成 (Streaming Generation)**：
   - 将召回内容转义单大括号后拼接成 `context`。
   - 动态拼接 `base_system_prompt`（强制声明“必须完全仅限于参考资料”）。
   - 通过 `chain_with_history.stream({"question": user_query_with_rule})` 将回答逐字 Yield 推送回前端，并在末尾附加引用资料来源 `{"type": "sources"}`。

### 1. 技术选型
*   **多路召回 (Hybrid Search)**：FAISS 稠密向量检索 + BM25 稀疏关键词检索 + Neo4j 知识图谱检索。
*   **精排模型 (Reranker)**：`bge-reranker-large`（部署于 GPU），将多路召回的异构结果进行交叉打分重排。
*   **生成模型后端 (LLM)**：
    *   支持动态切换：DeepSeek (API)、Qwen (vLLM 本地部署 / 阿里云 API)、ChatGPT、Ollama (qwen2.5 等轻量模型)。
*   **历史会话记忆**：Redis（`RedisChatMessageHistory`），为多轮对话提供低延迟上下文。

### 2. 参数配置
*   **多阶段召回参数**：
    *   **自适应融合召回**：不再固定召回 30 条。系统会通过意图启发式（正则探测）判断 Query 特征。若为精确名词查询（包含缩写等），则 BM25 召回量提升至 60 且权重设为 1.5；若是模糊描述，向量召回量提升至 60 且权重设为 1.5。
    *   **精排基数池**：融合后的轻量级候选集提升至 `Top-100`，再截断送入 GPU Reranker 进行细排。
*   **动态图检索深度 (Graph RAG)**：Cypher 查询采用两段式，先执行 `1` 跳检索（限 15 条），若结果少于 3 个，则自动降级触发 `2` 跳查询（引入 PageRank 简化版的度中心性重要度排序）。
*   **大模型生成温度**：
    *   普通 QA 生成 `temperature=0.1`（保障回答的严谨性和专业性）。

### 3. 优化策略
*   **意图识别前置与问题重构**：
    *   在进入处理流前，先用轻量级 Prompt 判断用户意图（`知识问答, 工具调用, 闲聊, 图谱查询`）。
    *   **历史对话动态裁剪**：对长对话仅保留最近 6 条（3轮）记录参与重构。
    *   由于多轮对话存在代词（如“它”、“这个服务”），若判定非“工具调用”意图，则结合历史将其交由 LLM 重构为**独立完整的问题 (Standalone Question)**。同时在重构时强制执行**实体链接**（如识别出 Nginx, MySQL 需明确标注）。
*   **Reranker 缓存机制**：
    *   引入内存级别的 `reranker_cache`，对高频相同问题（FAQ）直接跳过昂贵的 GPU 重排推理，实现毫秒级返回。为防止 Token 越界，还在构建索引和重排前加入了严格的 800 字符硬截断策略。
*   **知识图谱拓扑增强**：
    *   放弃用大模型慢速抽取，改用**预置运维词库的最大前向匹配（NER 模型化）**快速识别实体。
    *   将图谱结构化结果转化为**自然语言描述**（如：“在系统拓扑中，节点 A 与节点 B 存在【依赖】关系”），并赋予其 `0.95` 的极高初始置信度，方便文本 Reranker 对齐语义。
*   **动态 Prompt 与置信度输出**：
    *   根据检索 Top1 分数动态拼接 System Prompt（如 `<0.7` 提示综合推理，`>0.9` 要求严格遵循文档）。
    *   在生成回答的末尾，强制 LLM 输出引用的资料溯源以及回答置信度（高/中/低）。
*   **流式思考过程 (Thinking Chain)**：
    *   在前端 SSE（Server-Sent Events）流式输出中，不仅返回答案，还实时推送后台状态：“正在意图解析... -> 检索耗时 0.45s... -> Agent 工具调用成功...”。大幅降低用户的等待焦虑。（已修复内部异常时导致前端 JSON 流解析断裂的问题）。

### 4. 问题解决方法
*   **图谱实体提取失败或超时**：
    *   *问题*：依靠大模型动态提取查询中的实体，有时会因超时导致图检索流程中断。
    *   *解决*：设计了 Fallback 机制。若大模型提取失败，系统降级使用预置运维词库（Zabbix, Nginx, MySQL 等）进行正则匹配。
*   **Redis 宕机导致不可用**：
    *   *问题*：单机部署时 Redis 服务未启动，导致应用直接崩溃。
    *   *解决*：初始化时执行 `ping()` 检测，若失败则将 `use_redis` 设为 `False`，自动降级为内存 `ChatMessageHistory` 字典，保障服务最低可用性。

---

## 第三阶段：工程化与 Agent 集成 (Engineering and Agent Integration)

系统不仅仅是一个问答工具，更是一个能执行动作的运维智能体（Agent）。通过集成各类工具，打通了“知识解答”与“操作排查”的闭环。

### 核心代码执行流程
1. **工具配置与挂载**：
   - 启动时读取 `app/config/base.py` 中的 `ENABLED_TOOLS` 列表。
   - 在 `app/services/qa_service.py` 中，当 `enable_tools=True` 且存在可用工具时，通过 `app/services/agent_tools.py` 中的 `get_tools_by_names()` 动态加载工具（如 `query_metrics`, `execute_shell_command`）。
2. **智能体实例创建 (Agent Initialization)**：
   - 调用 `langchain.agents.create_tool_calling_agent(self.llm, tools, agent_prompt)` 绑定大模型与工具描述，形成具备函数调用（Function Calling）能力的 Agent。
   - 如果 LangChain 版本较低，系统会平滑触发 Fallback 机制，利用 LCEL 原生管道 `OpenAIToolsAgentOutputParser()` 手动拼接 Agent。
3. **ReAct 推理与链式执行 (AgentExecutor)**：
   - 初始化执行器：`AgentExecutor(agent=agent, tools=tools, max_iterations=3, return_intermediate_steps=True)`。
   - 调用 `agent_executor.invoke()` 传入用户问题与上下文。
   - 大模型在内部开启“思考-观察-行动”循环：
     - 若判定需要查状态，则大模型输出 ToolCall 指令。
     - 触发本地 `app/services/agent_tools.py` 中的 `@tool` 装饰器函数（如 `query_metrics("nginx")`）。
     - 将工具返回的结果（如 CPU: 85%）抛回给大模型作为 Observation，继续推理。
4. **思维链暴露 (CoT Streaming)**：
   - 获取执行器的 `intermediate_steps` 字段。
   - 遍历每一步 Action，通过 `yield json.dumps()` 以 Markdown 格式将“工具名称”、“输入参数”和“执行结果片段”实时推送到前端控制台，实现过程透明。
5. **最终结果提权**：
   - Agent 完成最终总结后，将完整的工具执行原始输出作为“优先级最高的相关资料（[1] 号文献）”注入到 Context 中，确保最终答案紧贴真实的系统运行状态。

### 1. 技术选型
*   **智能体架构**：全面引入 LangChain 的 `create_tool_calling_agent` 和 `AgentExecutor`，支持多步推理（ReAct Pattern）。
*   **工具集定义**：
    *   `query_metrics`：**【新增】**前置指标查询工具，用于在执行高危命令前获取服务健康度。
    *   `execute_shell_command`：执行受限的系统状态查询命令。
    *   `query_api_endpoint`：调用外部 REST API（如 Zabbix 监控大屏接口、K8s 状态接口）。
*   **应用部署**：Flask 框架，推荐使用 Gunicorn 多进程部署，静态资源统一由 Nginx 代理。

### 2. 参数配置
*   **工具开关配置**：通过环境变量 `ENABLED_TOOLS` 动态启停特定工具（例如关闭内网的 shell 权限）。
*   **AgentExecutor 迭代限制**：设置 `max_iterations=3`，防止 Agent 在处理复杂任务时陷入死循环。
*   **API/Shell 执行限制**：
    *   设置子进程执行超时时间 `timeout=10`，防止僵尸进程。
    *   API 响应截断 `[:2000]`，防止大规模 JSON 返回导致 LLM Context Overflow 报错。

### 3. 优化策略
*   **ReAct 多步规划与链式调用**：
    *   系统不再是“只能调用一次工具”的单向流。通过 ReAct 模式，Agent 能够根据上一个工具的返回结果，自主决定下一步动作。例如：“先执行 `query_metrics` 获取 Nginx 指标，发现 CPU 告警后，再决定调用 `execute_shell_command` 执行 `top` 命令排查进程”。
*   **CoT 思维链自省输出**：
    *   开启了 `return_intermediate_steps=True`，将 Agent 内部的推理过程（调用的工具名称、入参、执行片段）通过 Markdown 实时流式暴露给用户，极大地提升了系统的可解释性与调试效率。
*   **大模型后端的无缝解耦与兼容降级**：
    *   通过 `Config.ACTIVE_LLM` 的工厂模式配置，系统兼容 OpenAI 协议的 vLLM、Ollama 和 DeepSeek API。
    *   **架构降级保护**：当 LangChain 版本过旧不支持 `create_tool_calling_agent` 时，系统会自动 Fallback 到基于 LCEL 拼接的原生 `OpenAIToolsAgentOutputParser`；当完全不支持 Agent 时，平滑降级为普通知识问答模式，保证系统永不宕机。

### 4. 问题解决方法
*   **高危命令执行漏洞**：
    *   *问题*：如果用户通过 Prompt 注入（如：“请帮我执行 rm -rf /”），Agent 可能会盲目执行。
    *   *解决*：在 `execute_shell_command` 装饰器内部设置硬编码黑名单 `forbidden_words = ['rm ', 'reboot', 'shutdown', 'mkfs', '>']`，在工具层进行第一道物理拦截。
*   **模型循环调用与死锁**：
    *   *问题*：某些弱模型在工具返回结果不理想时，会反复尝试调用该工具导致死循环。
    *   *解决*：利用 `AgentExecutor` 的 `max_iterations` 参数设定硬性阈值，一旦超出步数即强制中断并返回当前总结。