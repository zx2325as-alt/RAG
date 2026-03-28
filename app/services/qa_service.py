import os
import hashlib
import json
import redis
import logging
from datetime import timedelta
from app.services.knowledge_base_service import KnowledgeBaseService
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.config import Config
from app.db import db
from app.db.models import QACache, QueryLog
from datetime import datetime

logger = logging.getLogger(__name__)

class QAService:
    def __init__(self):
        self.kb_service = KnowledgeBaseService()
        # 尝试连接 Redis，如果失败则在单机版中禁用缓存
        try:
            self.redis_client = redis.from_url(Config.REDIS_URL)
            self.redis_client.ping()
            self.use_redis = True
        except Exception as e:
            logger.warning(f"Redis not available ({e}). Caching disabled.")
            self.use_redis = False
            
        # Initialize LLM 根据配置动态切换大模型
        self.initialize_llm()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位严谨的湖北移动智能运维专家。
你的任务是**优先基于提供的参考资料**，为用户提供准确、专业的解答。

【回答核心原则 - 绝对遵守】：
1. 工具结果最高优先级：如果 `<参考资料>` 中包含了【工具执行结果】，你必须直接提取该结果中的核心数据或状态向用户汇报，此时**绝对不要**再说“资料中未提及相关内容”，也**不要**输出【非基于文档回答】的标记。
2. 忠于检索内容：在没有工具结果但有参考资料时，你是一个资料整合者，所有回答必须来源于 `<参考资料>`。
3. 兜底机制：只有在既没有【工具执行结果】，也没有相关的 `<参考资料>` 时，才允许利用你的预训练知识给出兜底答案，且此时必须在显著位置注明“【非基于文档回答】”。

【排版要求】：
- 结构化输出：对于故障排查或操作指南，请分步骤（如：1.现象分析；2.可能原因；3.排查步骤）进行条理清晰的解答。
- 详尽专业：用运维领域的专业语言对资料进行归纳和总结。如果参考资料提供了多个方案，请全面列出并说明各自的适用场景。
- 内联引用标记：在基于文档回答事实性内容或数据时，必须在句子末尾使用 [1], [2] 这种角标格式标注出对应的参考资料序号，以增强回答的可信度。

<参考资料>
{context}
</参考资料>
"""),
            MessagesPlaceholder(variable_name="history"),
            ("user", "我的问题是：{question}\n优先根据参考资料给出解答。若资料中未提及相关内容，请提供兜底解答并标记为“非基于文档回答”。")
        ])
        
        # 构建带历史记录的对话 Chain
        self.chain = self.prompt | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    def get_session_history(self, session_id: str):
        """获取 Redis 中的对话历史记录"""
        # 如果未配置 Redis，使用简单的内存字典（演示用）
        if self.use_redis:
            return RedisChatMessageHistory(session_id, url=Config.REDIS_URL)
        else:
            from langchain_community.chat_message_histories import ChatMessageHistory
            if not hasattr(self, 'memory_histories'):
                self.memory_histories = {}
            if session_id not in self.memory_histories:
                self.memory_histories[session_id] = ChatMessageHistory()
            return self.memory_histories[session_id]

    def initialize_llm(self):
        active_llm = Config.ACTIVE_LLM.lower()
        logger.info(f"Initializing LLM with backend: {active_llm}")
        
        if active_llm == 'deepseek':
            self.llm = ChatOpenAI(
                base_url=Config.DEEPSEEK_API_URL.replace('/chat/completions', ''), 
                api_key=Config.DEEPSEEK_API_KEY,
                model=Config.DEEPSEEK_MODEL_NAME,
                temperature=0.3,
                max_retries=1,
                timeout=60
            )
        elif active_llm == 'chatgpt':
            self.llm = ChatOpenAI(
                api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
                model='gpt-3.5-turbo',
                temperature=0.3,
                max_retries=1,
                timeout=60
            )
        elif active_llm == 'ollama':
            from langchain_community.chat_models import ChatOllama
            self.llm = ChatOllama(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_MODEL_NAME,
                temperature=0.3
            )
        elif active_llm == 'vllm':
            # vLLM 兼容 OpenAI 接口，需要将 /chat/completions 后缀去掉，保留到 /v1 级别
            base_url = Config.VLLM_API_URL
            if base_url.endswith('/chat/completions'):
                base_url = base_url.replace('/chat/completions', '')
                
            self.llm = ChatOpenAI(
                base_url=base_url, 
                api_key=Config.VLLM_API_KEY,
                model=Config.VLLM_MODEL_NAME,
                temperature=0.3,
                max_retries=1,
                timeout=60
            )
        else: # 默认使用 qwen 或者其它在线模型（动态传入的名字）
            # 检查是否是在线模型列表中的模型
            is_online = any(m['id'] == active_llm for m in Config.ONLINE_QUERY_MODELS)
            
            if is_online and active_llm == 'qwen':
                # 专门处理通义千问官方API (阿里云百炼)
                self.llm = ChatOpenAI(
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                    api_key=os.getenv('DASHSCOPE_API_KEY', Config.QWEN_API_KEY),
                    model="qwen-plus", # 默认使用 qwen-plus
                    temperature=0.3,
                    max_retries=1,
                    timeout=60
                )
            else:
                self.llm = ChatOpenAI(
                    base_url=Config.QWEN_API_URL.replace('/chat/completions', ''), 
                    api_key=Config.QWEN_API_KEY,
                    model=Config.QWEN_MODEL_NAME,
                    temperature=0.3,
                    max_retries=1,
                    timeout=60
                )
        
        # 重新绑定 chain
        if hasattr(self, 'prompt'):
            self.chain = self.prompt | self.llm
            self.chain_with_history = RunnableWithMessageHistory(
                self.chain,
                self.get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )

    def get_cache_key(self, question):
        return hashlib.md5(question.encode('utf-8')).hexdigest()

    def stream_answer_question(self, question, user_id="anonymous", db_names=['default'], enable_tools=True):
        start_time = datetime.utcnow()
        cache_key = self.get_cache_key(question + "_" + "_".join(db_names) + "_" + str(enable_tools))
        
        # 1. Check Redis Cache
        if self.use_redis:
            try:
                cached_response = self.redis_client.get(cache_key)
                if cached_response:
                    logger.info("Hit Redis Cache")
                    data = json.loads(cached_response)
                    # 模拟流式输出缓存内容
                    yield json.dumps({"type": "chunk", "content": data["answer"]}) + "\n"
                    yield json.dumps({"type": "sources", "sources": data["sources"]}) + "\n"
                    return
            except Exception as e:
                logger.error(f"Redis get error: {e}")

        # 1.5 改写用户问题 (Standalone Question)
        search_query = question
        yield json.dumps({"type": "thought", "content": "- [x] 收到用户请求，正在解析意图...\n"}) + "\n"
        
        try:
            history_obj = self.get_session_history(user_id)
            if len(history_obj.messages) > 0:
                yield json.dumps({"type": "thought", "content": "- [x] 检测到多轮对话，正在结合历史上下文重构问题...\n"}) + "\n"
                from langchain_core.messages import SystemMessage, HumanMessage
                history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history_obj.messages[-4:]])
                rewrite_prompt = f"""你是一个智能运维助手。请根据用户的历史对话和当前提问，将当前提问重写为一个独立、完整且表意清晰的搜索查询（Query）。
如果当前提问是一个系统命令（如 ping, df -h），请不要重写，直接返回原命令。
如果当前提问已经很完整，也直接返回原提问。

历史对话：
{history_text}

当前提问：{question}
重写后的完整提问："""
                rewrite_response = self.llm.invoke([HumanMessage(content=rewrite_prompt)])
                search_query = rewrite_response.content.strip()
                logger.info(f"Original Query: {question} -> Rewritten Query: {search_query}")
                yield json.dumps({"type": "thought", "content": f"- [x] 问题重构完成: `{search_query}`\n"}) + "\n"
            else:
                yield json.dumps({"type": "thought", "content": "- [x] 独立问题，跳过重构步骤。\n"}) + "\n"
        except Exception as e:
            logger.error(f"Rewrite Query Error: {e}")
            if "Connection error" in str(e) or "Connection refused" in str(e):
                yield json.dumps({"type": "chunk", "content": "\n抱歉，连接大模型服务失败，请检查 LLM 服务或 Redis 状态。"}) + "\n"
                return
            yield json.dumps({"type": "thought", "content": "- [!] 问题重构失败，将使用原始问题继续。\n"}) + "\n"

        # 增加思考过程展示：开始检索
        yield json.dumps({"type": "thought", "content": "- [ ] 正在执行多路知识召回 (向量库检索 / 关键词 BM25 匹配 / 知识图谱遍历)...\n"}) + "\n"

        # 3. Retrieval 使用改写后的完整问题进行检索，支持多库
        import time
        t1 = time.time()
        results = self.kb_service.search(search_query, top_k=5, db_names=db_names)
        t2 = time.time()
        
        # 将检索到的具体库、数量、耗时展示出来
        search_details = f"耗时 {round(t2-t1, 2)}s"
        if results:
            docs_info = "\n  - " + "\n  - ".join([f"匹配度: {round(score, 3)} | 来源: {doc.metadata.get('doc_name', '未知')}" for doc, score in results[:3]])
            if len(results) > 3:
                docs_info += f"\n  - ...等共 {len(results)} 条片段"
        else:
            docs_info = "\n  - 未检索到强相关内容"

        # 增加思考过程展示：检索完成，评估工具
        if enable_tools:
            yield json.dumps({"type": "thought", "content": f"- [x] 检索完成 ({search_details})，召回并融合重排了 {len(results)} 条知识片段:{docs_info}\n- [ ] 正在评估是否需要调用外部工具获取实时状态...\n"}) + "\n"
        else:
            yield json.dumps({"type": "thought", "content": f"- [x] 检索完成 ({search_details})，召回并融合重排了 {len(results)} 条知识片段:{docs_info}\n- [ ] 外部工具调用已禁用，直接进入知识整合阶段...\n"}) + "\n"
        
        # 3.5 Agent 工具调用 (Function Calling)
        # 判断大模型是否需要调用工具获取实时数据
        tool_context = ""
        if enable_tools:
            try:
                from app.services.agent_tools import get_tools_by_names, AVAILABLE_TOOLS
                
                # 从配置中动态加载工具列表
                enabled_tool_names = getattr(Config, 'ENABLED_TOOLS', ['execute_shell_command', 'query_api_endpoint'])
                tools = get_tools_by_names(enabled_tool_names)
                
                if tools:
                    # 绑定工具到大模型
                    llm_with_tools = self.llm.bind_tools(tools)
                    # 强化 Prompt，明确如果问题是命令（如 ping）必须直接调用对应的执行工具
                    tool_prompt = f"""分析以下用户的输入。如果它是一个系统命令（例如 'ping xxx'，'df -h' 等），或者需要获取实时的主机/系统状态、外部API数据，
    你**必须**调用相应的工具去获取结果。可用的工具包括：{', '.join(enabled_tool_names)}。
    如果只是普通的知识问答，不需要调用工具。
    用户输入：{search_query}"""
                    
                    tool_msg = llm_with_tools.invoke([("system", "你是一个能够调用工具的智能助手。"), ("user", tool_prompt)])
                    if tool_msg.tool_calls:
                        yield json.dumps({"type": "thought", "content": "- [x] 问题涉及实时状态，触发外部工具调用逻辑。\n"}) + "\n"
                        for tool_call in tool_msg.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                            yield json.dumps({"type": "thought", "content": f"- [ ] 正在执行工具: `{tool_name}` ...\n"}) + "\n"
                            
                            # 动态调用对应工具
                            if tool_name in AVAILABLE_TOOLS:
                                tool_func = AVAILABLE_TOOLS[tool_name]
                                res = tool_func.invoke(tool_args)
                                tool_context += f"【{tool_name} 执行结果】:\n{res}\n\n"
                                yield json.dumps({"type": "thought", "content": f"- [x] 工具 `{tool_name}` 执行成功，已获取实时数据。\n"}) + "\n"
                                
                        yield json.dumps({"type": "thought", "content": "- [x] 所有前置任务完成，开始整合上下文并生成最终分析报告...\n\n"}) + "\n"
                    else:
                        yield json.dumps({"type": "thought", "content": "- [x] 评估完毕：无需调用外部工具。\n- [x] 所有前置任务完成，开始整合知识生成最终分析报告...\n\n"}) + "\n"
                else:
                    yield json.dumps({"type": "thought", "content": "- [x] 未配置外部工具，跳过评估。\n- [x] 所有前置任务完成，开始整合知识生成最终分析报告...\n\n"}) + "\n"
            except Exception as e:
                logger.error(f"Tool Calling Error: {e}")
                yield json.dumps({"type": "thought", "content": "- [!] 工具调用或评估时发生错误，已跳过。\n- [x] 开始整合可用知识生成分析报告...\n\n"}) + "\n"
        else:
            # 工具被禁用，直接输出完成信息
            yield json.dumps({"type": "thought", "content": "- [x] 所有前置任务完成，开始整合知识生成最终分析报告...\n\n"}) + "\n"

        # 告诉前端思考结束，准备输出正文
        yield json.dumps({"type": "thought_end"}) + "\n"

        # 4. Prepare Context
        context_parts = []
        sources = []
        
        # 为了保证 LLM 引用的 [1], [2] 和前端展示的来源一致，我们需要统一计数
        source_index = 1
        
        # 优先将工具调用结果作为最强关联资料放入 context
        if tool_context:
            context_parts.append(f"资料 [{source_index}] (【工具执行结果】请优先根据此结果回答用户):\n{tool_context}")
            sources.append({
                "content": tool_context[:100] + "...",
                "doc_id": "tool_execution",
                "doc_name": "系统实时执行结果",
                "score": 1.0
            })
            source_index += 1
            
        for doc, score in results:
            # 将序号一并编入 Context 供 LLM 参考
            context_parts.append(f"资料 [{source_index}]:\n{doc.page_content}")
            doc_name = doc.metadata.get("doc_name", "Unknown")
            
            # 如果元数据里是 Unknown，尝试从数据库里查一次
            if doc_name == "Unknown":
                try:
                    from app.db import db
                    from app.db.models import Document
                    from flask import current_app
                    with current_app.app_context():
                        doc_id = doc.metadata.get("doc_id")
                        if doc_id:
                            db_doc = db.session.query(Document).filter_by(doc_id=doc_id).first()
                            if db_doc:
                                doc_name = db_doc.doc_name
                except Exception:
                    pass

            sources.append({
                "content": doc.page_content[:100] + "...",
                "doc_id": doc.metadata.get("doc_id"),
                "doc_name": doc_name,
                "score": float(score)
            })
            source_index += 1
            
        context = "\n\n".join(context_parts)
        
        # 4. Generate Answer via Stream
        full_answer = ""
        try:
            # 使用 LangChain 的 stream 方法，并传入 session_id
            for chunk in self.chain_with_history.stream(
                {"context": context, "question": question},
                config={"configurable": {"session_id": user_id}}
            ):
                content = chunk.content
                if content:
                    full_answer += content
                    yield json.dumps({"type": "chunk", "content": content}) + "\n"
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            error_msg = "抱歉，生成答案时出现错误。请检查LLM服务状态。"
            yield json.dumps({"type": "chunk", "content": error_msg}) + "\n"
            full_answer = error_msg
            
        # 发送引用来源
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        
        # 5. Save to Cache
        result = {
            "answer": full_answer,
            "sources": sources
        }
        if self.use_redis and full_answer and "错误" not in full_answer:
            try:
                self.redis_client.setex(cache_key, timedelta(seconds=3), json.dumps(result))
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        # 6. Log Query (使用 app.app_context 确保在生成器线程中也能正常记录数据库)
        try:
            latency = (datetime.utcnow() - start_time).total_seconds()
            from app.db import db
            from app.db.models import QueryLog
            # 获取当前 Flask app 实例
            from flask import current_app
            with current_app.app_context():
                log = QueryLog(
                    user_id=user_id,
                    query=question,
                    response=full_answer,
                    latency=latency
                )
                db.session.add(log)
                db.session.commit()
        except Exception as e:
            logger.error(f"DB Log error: {e}")

    def stream_reanalyze_question(self, question, previous_answer, score, user_id="anonymous"):
        """处理低于4分的重新分析工作流"""
        yield json.dumps({"type": "chunk", "content": f"\n\n**[系统检测到评分 {score} 分，触发深度重新分析工作流...]**\n\n"}) + "\n"
        
        # 1. 扩大检索范围 (从 5 扩大到 15)
        results = self.kb_service.search(question, top_k=15)
        
        context_parts = []
        sources = []
        for i, (doc, score) in enumerate(results):
            context_parts.append(f"资料 [{i+1}]:\n{doc.page_content}")
            doc_name = doc.metadata.get("doc_name", "Unknown")
            
            if doc_name == "Unknown":
                try:
                    from app.db import db
                    from app.db.models import Document
                    from flask import current_app
                    with current_app.app_context():
                        doc_id = doc.metadata.get("doc_id")
                        if doc_id:
                            db_doc = db.session.query(Document).filter_by(doc_id=doc_id).first()
                            if db_doc:
                                doc_name = db_doc.doc_name
                except Exception:
                    pass
                    
            sources.append({
                "content": doc.page_content[:100] + "...",
                "doc_id": doc.metadata.get("doc_id"),
                "doc_name": doc_name,
                "score": float(score)
            })
            
        context = "\n\n".join(context_parts)
        
        # 2. 动态调整 Prompt (根据分数不同，优化策略不同)
        if int(score) <= 2:
            strategy_prompt = """
用户指出上一轮回答存在严重错误或未能解决问题。
请进行自我反思与交叉验证：仔细核对以下所有提供的参考资料。
1. 如果发现资料之间存在冲突，请明确指出。
2. 如果提供的资料中完全没有任何与问题相关的解决办法或线索（即目前资料中未提及相关内容），请利用你的预训练知识提供一个兜底的解答或排查方向，但**必须**做出明确标记为“【非基于文档回答】”，并建议用户结合实时监控系统进行检查。
"""
        else:
            strategy_prompt = "用户认为上一次回答不够完美。请在参考资料的基础上，补充更多细节，重点说明各个方案的优缺点和适用场景，并确保证明依据充分。如果资料不足，可补充带有“【非基于文档回答】”标记的兜底解答。"
            
        reanalyze_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是一位专家级的故障诊断顾问。
{strategy_prompt}
请严格基于以下**扩充后的参考资料**进行深度分析：

<参考资料>
{{context}}
</参考资料>
"""),
            MessagesPlaceholder(variable_name="history"),
            ("user", "针对我的问题：{question}\n请给出更加深入、全面、专业的重新分析报告。")
        ])
        
        chain = reanalyze_prompt | self.llm
        reanalyze_chain_with_history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        
        try:
            for chunk in reanalyze_chain_with_history.stream(
                {"context": context, "question": question},
                config={"configurable": {"session_id": user_id}}
            ):
                content = chunk.content
                if content:
                    yield json.dumps({"type": "chunk", "content": content}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "chunk", "content": f"重新分析失败: {str(e)}"}) + "\n"
            
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
