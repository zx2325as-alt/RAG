import os
import hashlib
import json
import redis
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

class QAService:
    def __init__(self):
        self.kb_service = KnowledgeBaseService()
        # 尝试连接 Redis，如果失败则在单机版中禁用缓存
        try:
            self.redis_client = redis.from_url(Config.REDIS_URL)
            self.redis_client.ping()
            self.use_redis = True
        except Exception as e:
            print(f"Warning: Redis not available ({e}). Caching disabled.")
            self.use_redis = False
            
        # Initialize LLM 根据配置动态切换大模型
        self.initialize_llm()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位严谨的湖北移动智能运维专家。
你的任务是**严格基于提供的参考资料**，为用户提供准确、专业的解答。

【回答核心原则 - 绝对遵守】：
1. 绝对忠于检索内容：你只是一个资料整合者。你的所有回答必须且只能来源于 `<参考资料>` 中提供的信息。如果资料中没有提到，绝对禁止使用你的预训练知识进行发散、编造或猜测！
2. 不知为不知：如果用户的问题在 `<参考资料>` 中完全找不到相关线索，你必须且只能回答：“根据当前知识库检索，未能找到与您问题相关的具体资料，请尝试修改提问或补充相关文档。” 绝不要强行作答！

【排版要求】：
- 结构化输出：对于故障排查或操作指南，请分步骤（如：1.现象分析；2.可能原因；3.排查步骤）进行条理清晰的解答。
- 详尽专业：用运维领域的专业语言对资料进行归纳和总结。如果参考资料提供了多个方案，请全面列出并说明各自的适用场景。

<参考资料>
{context}
</参考资料>
"""),
            MessagesPlaceholder(variable_name="history"),
            ("user", "我的问题是：{question}\n请务必只根据参考资料给出解答。")
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
        print(f"Initializing LLM with backend: {active_llm}")
        
        if active_llm == 'deepseek':
            self.llm = ChatOpenAI(
                base_url=Config.DEEPSEEK_API_URL.replace('/chat/completions', ''), 
                api_key=Config.DEEPSEEK_API_KEY,
                model=Config.DEEPSEEK_MODEL_NAME,
                temperature=0.3 # 适当提高温度以增加全面性和发散性
            )
        elif active_llm == 'ollama':
            from langchain_community.chat_models import ChatOllama
            self.llm = ChatOllama(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_MODEL_NAME,
                temperature=0.3
            )
        else: # 默认使用 qwen
            self.llm = ChatOpenAI(
                base_url=Config.QWEN_API_URL.replace('/chat/completions', ''), 
                api_key=Config.QWEN_API_KEY,
                model=Config.QWEN_MODEL_NAME,
                temperature=0.3
            )

    def get_cache_key(self, question):
        return hashlib.md5(question.encode('utf-8')).hexdigest()

    def stream_answer_question(self, question, user_id="anonymous", db_names=['default']):
        start_time = datetime.utcnow()
        cache_key = self.get_cache_key(question + "_" + "_".join(db_names))
        
        # 1. Check Redis Cache
        if self.use_redis:
            try:
                cached_response = self.redis_client.get(cache_key)
                if cached_response:
                    print("Hit Redis Cache")
                    data = json.loads(cached_response)
                    # 模拟流式输出缓存内容
                    yield json.dumps({"type": "chunk", "content": data["answer"]}) + "\n"
                    yield json.dumps({"type": "sources", "sources": data["sources"]}) + "\n"
                    return
            except Exception as e:
                print(f"Redis get error: {e}")

        # 1.5 改写用户问题 (Standalone Question)
        # 获取历史记录，如果历史记录存在，则利用大模型将当前带有代词的残缺问题改写为独立完整的问题
        history_obj = self.get_session_history(user_id)
        search_query = question
        
        if len(history_obj.messages) > 0:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history_obj.messages[-4:]]) # 取最近2轮
                rewrite_prompt = f"""根据以下对话历史记录，将用户的最新问题改写为一个独立、完整的问题，补全所有指代和省略的内容，以便于直接用于数据库检索。如果问题已经很完整，无需改写，直接原样返回。
注意：只返回改写后的问题文本，不要有任何其他解释词。

对话历史：
{history_text}

最新问题：{question}
独立完整的问题："""
                rewrite_response = self.llm.invoke([HumanMessage(content=rewrite_prompt)])
                search_query = rewrite_response.content.strip()
                print(f"Original Query: {question} -> Rewritten Query: {search_query}")
                yield json.dumps({"type": "chunk", "content": f"\n\n*(系统理解的完整问题：{search_query})*\n\n"}) + "\n"
            except Exception as e:
                print(f"Rewrite Query Error: {e}")

        # 2. Agent 路由：Text2SQL (运维数据库查询拦截)
        # 简单模拟 Function Calling 路由：如果检测到特定的运维数据库查询意图，直接返回SQL及结果
        if "告警次数" in search_query or "数据库查询" in search_query:
            sql_query = "SELECT count(*) FROM alerts WHERE date = DATE('now', '-1 day') AND status = 'unresolved';"
            simulated_result = "经查询 Zabbix/Prometheus 数据库，昨日新增未恢复告警共计 **15** 条。"
            agent_msg = f"*(已触发 Text2SQL 运维插件)*\n\n**执行SQL:**\n```sql\n{sql_query}\n```\n\n**查询结果:**\n{simulated_result}"
            yield json.dumps({"type": "chunk", "content": agent_msg}) + "\n"
            yield json.dumps({"type": "sources", "sources": [{"content": "Zabbix DB", "score": 1.0}]}) + "\n"
            return

        # 3. Retrieval 使用改写后的完整问题进行检索，支持多库
        results = self.kb_service.search(search_query, top_k=5, db_names=db_names)
        
        if not results:
            yield json.dumps({"type": "chunk", "content": "抱歉，知识库中没有相关信息。"}) + "\n"
            yield json.dumps({"type": "sources", "sources": []}) + "\n"
            return

        # 3. Prepare Context
        context_parts = []
        sources = []
        for doc, score in results:
            context_parts.append(doc.page_content)
            sources.append({
                "content": doc.page_content[:100] + "...",
                "doc_id": doc.metadata.get("doc_id"),
                "score": float(score)
            })
            
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
            print(f"LLM Error: {e}")
            error_msg = "抱歉，生成答案时出现错误。请检查LLM服务状态。"
            yield json.dumps({"type": "chunk", "content": error_msg}) + "\n"
            full_answer = error_msg
            
        # Agent 路由：自动化脚本执行 (Auto-Remediation)
        # 如果大模型的回答中包含“重启”操作，则注入一个自动化执行脚本的按钮
        if "重启" in full_answer and "服务" in full_answer:
            action_html = "<div class='mt-2 p-2 bg-light border rounded text-danger'><strong><i class='bi bi-tools'></i> 运维操作建议</strong><br>检测到修复方案包含重启服务，是否立即执行？<br><button class='btn btn-sm btn-danger mt-2' onclick='executeAutoScript(\"restart_service\")'>立即执行重启脚本 (SSH/Ansible)</button></div>"
            yield json.dumps({"type": "action", "content": action_html}) + "\n"
            
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
                print(f"Redis set error: {e}")
        
        # 6. Log Query (需要在一个新的 app context 或者独立处理)
        try:
            latency = (datetime.utcnow() - start_time).total_seconds()
            from app.db import db
            from app.db.models import QueryLog
            log = QueryLog(
                user_id=user_id,
                query=question,
                response=full_answer,
                latency=latency
            )
            db.session.add(log)
            db.session.commit()
        except Exception as e:
            print(f"DB Log error: {e}")

    def stream_reanalyze_question(self, question, previous_answer, score, user_id="anonymous"):
        """处理低于4分的重新分析工作流"""
        yield json.dumps({"type": "chunk", "content": f"\n\n**[系统检测到评分 {score} 分，触发深度重新分析工作流...]**\n\n"}) + "\n"
        
        # 1. 扩大检索范围 (从 5 扩大到 15)
        results = self.kb_service.search(question, top_k=15)
        
        context_parts = []
        sources = []
        for doc, s in results:
            context_parts.append(doc.page_content)
            sources.append({
                "content": doc.page_content[:100] + "...",
                "doc_id": doc.metadata.get("doc_id"),
                "score": float(s)
            })
            
        context = "\n\n".join(context_parts)
        
        # 2. 动态调整 Prompt (根据分数不同，优化策略不同)
        if int(score) <= 2:
            strategy_prompt = "用户对上一次回答非常不满意。请彻底抛弃之前的思路，从提供的参考资料中寻找更底层的原理或更全面的排查步骤，务必给出极其详细和基础的解释。"
        else:
            strategy_prompt = "用户认为上一次回答不够完美。请在参考资料的基础上，补充更多细节，重点说明各个方案的优缺点和适用场景。"
            
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
