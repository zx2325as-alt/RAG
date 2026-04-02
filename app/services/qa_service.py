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
        print(f"Initializing LLM with backend: {active_llm}")
        
        if active_llm == 'deepseek':
            self.llm = ChatOpenAI(
                base_url=Config.DEEPSEEK_API_URL.replace('/chat/completions', ''), 
                api_key=Config.DEEPSEEK_API_KEY,
                model=Config.DEEPSEEK_MODEL_NAME,
                temperature=0.1,
                max_retries=1,
                timeout=60
            )
        elif active_llm == 'chatgpt':
            self.llm = ChatOpenAI(
                api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
                model='gpt-3.5-turbo',
                temperature=0.1,
                max_retries=1,
                timeout=60
            )
        elif active_llm == 'ollama':
            from langchain_community.chat_models import ChatOllama
            self.llm = ChatOllama(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_MODEL_NAME,
                temperature=0.1
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
                temperature=0.1,
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
                    temperature=0.1,
                    max_retries=1,
                    timeout=60
                )
            else:
                self.llm = ChatOpenAI(
                    base_url=Config.QWEN_API_URL.replace('/chat/completions', ''), 
                    api_key=Config.QWEN_API_KEY,
                    model=Config.QWEN_MODEL_NAME,
                    temperature=0.1,
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
    
    def _check_needs_rewrite(self, question, history_messages):
        """
        启发式判断是否需要问题重构
        返回True表示需要重构，False表示可以直接使用原问题
        """
        if not history_messages:
            return False
        
        # 指代词列表
        pronouns = ['它', '他', '她', '这个', '那个', '这些', '那些', '这里', '那里', '这样', '那样']
        
        # 如果包含指代词，需要重构
        if any(p in question for p in pronouns):
            return True
        
        # 如果问题太短（<5字），可能需要上下文
        # 注意：调用处已检查历史消息，只有存在历史消息时才会执行到这里
        if len(question) < 5:
            return True
        
        # 如果包含"为什么"、"怎么办"等，且历史中有相关内容
        follow_up_keywords = ['为什么', '怎么办', '然后呢', '还有呢', '除此之外']
        if any(kw in question for kw in follow_up_keywords):
            return True
        
        return False
    
    def _post_process_rewrite(self, original, rewritten, history_text=None):
        """
        问题重构后处理：防止过度发散
        """
        # 如果改写结果过长（超过原问题5倍），可能过度发散
        if len(rewritten) > len(original) * 5:
            print(f"Warning: Rewritten query too long, falling back to original. Original: {original}, Rewritten: {rewritten}")
            return original
        
        # 如果改写结果包含原问题中没有的关键词（可能是幻觉）
        original_words = set(original.lower().split())
        rewritten_words = set(rewritten.lower().split())
        new_words = list(rewritten_words - original_words)
        
        # 在检查新词比例之前，过滤掉来自历史对话的词
        if history_text:
            history_words = set(history_text.lower().split())
            new_words = [w for w in new_words if w not in history_words]
        
        # 如果新增词汇过多（超过70%），可能过度发散
        if len(new_words) > len(original_words) * 0.7 and len(original_words) > 3:
            print(f"Warning: Rewritten query has too many new words, using original.")
            return original
        
        return rewritten

    def stream_answer_question(self, question, user_id="anonymous", db_names=['default'], enable_tools=True):
        start_time = datetime.utcnow()
        cache_key = self.get_cache_key(question + "_" + "_".join(db_names) + "_" + str(enable_tools))
        
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

        # ========== 思考过程开始 ==========
        yield json.dumps({"type": "chunk", "content": "> 🧠 **思考过程**\n"}) + "\n"
        yield json.dumps({"type": "chunk", "content": "> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"}) + "\n"
        
        # 步骤1: 接收请求
        yield json.dumps({"type": "chunk", "content": "> **步骤 1/6: 接收用户请求** ✅\n"}) + "\n"
        yield json.dumps({"type": "chunk", "content": f">   📌 原始问题: `{question}`\n"}) + "\n"
        yield json.dumps({"type": "chunk", "content": f">   📌 查询范围: {', '.join(db_names)}\n"}) + "\n"
        yield json.dumps({"type": "chunk", "content": f">   📌 工具模式: {'开启' if enable_tools else '关闭'}\n"}) + "\n"
        
        # 步骤2: 历史上下文处理
        yield json.dumps({"type": "chunk", "content": "> **步骤 2/6: 历史上下文处理**\n"}) + "\n"
        
        search_query = question
        try:
            history_obj = self.get_session_history(user_id)
            messages = history_obj.messages
            
            # 历史消息干扰：只保留最近 2-3 轮，过滤掉工具调用等中间步骤，只保留最终的 user 和 assistant 消息
            filtered_messages = []
            for msg in messages:
                if msg.type in ['human', 'ai'] and getattr(msg, 'name', None) is None:
                    filtered_messages.append(msg)
            
            # 保留最近 4 条消息（2轮对话）
            if len(filtered_messages) > 4:
                filtered_messages = filtered_messages[-4:]
            
            yield json.dumps({"type": "chunk", "content": f">   📜 历史消息数: {len(messages)} 条\n"}) + "\n"
            yield json.dumps({"type": "chunk", "content": f">   📜 有效上下文: {len(filtered_messages)} 条\n"}) + "\n"
            
            # 问题重构优化：先进行启发式判断，减少不必要的LLM调用
            needs_rewrite = self._check_needs_rewrite(question, filtered_messages)
            
            if needs_rewrite and len(filtered_messages) > 0 and not enable_tools:
                yield json.dumps({"type": "chunk", "content": ">   🔄 执行问题重构...\n"}) + "\n"
                history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in filtered_messages])
                
                # 显示历史上下文摘要
                history_summary = " | ".join([f"{msg.type[:1].upper()}: {msg.content[:30]}..." for msg in filtered_messages[-2:]])
                yield json.dumps({"type": "chunk", "content": f">   💭 上下文摘要: {history_summary}\n"}) + "\n"
                
                # 优化后的重构提示词，增加意图保持约束
                rewrite_prompt = f"""请根据以下对话历史记录，将用户的最新问题改写为一个独立、完整且没有歧义的问题。

【改写要求 - 严格遵守】：
1. **仅补全指代词**：将"它"、"这个"、"那个"等指代词替换为上文提到的具体实体名称。
2. **保持原始意图**：严禁添加原问题中没有的信息、严禁扩展问题范围、严禁改变问题类型。
3. **系统指令原样返回**：如果问题是明确的系统命令（如ping、curl、kubectl等），直接原样返回。
4. **清晰问题无需改写**：如果问题已经完整清晰（不含指代词），直接原样返回。
5. **输出格式**：只返回改写后的纯文本问题，不要有任何解释、引号或额外内容。

【对话历史】：
{history_text}

【最新问题】：{question}

【独立完整的问题】："""
                from langchain_core.messages import HumanMessage
                try:
                    rewrite_response = self.llm.invoke([HumanMessage(content=rewrite_prompt)])
                    search_query = rewrite_response.content.strip()
                except Exception as e:
                    logging.warning(f"问题重构LLM调用失败，使用原始问题: {e}")
                    search_query = question
                
                # 后处理：确保改写不会过度发散
                original_query = search_query
                search_query = self._post_process_rewrite(question, search_query, history_text)
                
                if search_query != original_query:
                    yield json.dumps({"type": "chunk", "content": ">   ⚠️ 重构结果被后处理修正\n"}) + "\n"
                
                if search_query != question:
                    yield json.dumps({"type": "chunk", "content": f">   ✅ 重构完成\n"}) + "\n"
                    yield json.dumps({"type": "chunk", "content": f">      原问题: `{question}`\n"}) + "\n"
                    yield json.dumps({"type": "chunk", "content": f">      重构后: `{search_query}`\n"}) + "\n"
                else:
                    yield json.dumps({"type": "chunk", "content": ">   ✅ 问题已清晰，无需重构\n"}) + "\n"
            else:
                reason = "无历史上下文" if len(filtered_messages) == 0 else "问题已完整清晰"
                yield json.dumps({"type": "chunk", "content": f">   ⏭️ 跳过重构 ({reason})\n"}) + "\n"
        except Exception as e:
            print(f"Rewrite Query Error: {e}")
            yield json.dumps({"type": "chunk", "content": f">   ❌ 重构失败: {str(e)}\n"}) + "\n"
            yield json.dumps({"type": "chunk", "content": ">   ⏭️ 使用原始问题继续\n"}) + "\n"

        # 步骤3: 查询意图分析
        yield json.dumps({"type": "chunk", "content": "> **步骤 3/6: 查询意图分析**\n"}) + "\n"
        intent = self.kb_service._analyze_query_intent(search_query)
        intent_type = "精确查询" if intent['is_exact_query'] else ("故障排查" if intent['is_troubleshooting'] else "语义查询")
        yield json.dumps({"type": "chunk", "content": f">   🔍 识别意图: {intent_type}\n"}) + "\n"
        yield json.dumps({"type": "chunk", "content": f">   ⚖️ 检索权重: 向量({intent['weights']['vector']}) | BM25({intent['weights']['bm25']}) | 图谱({intent['weights']['graph']})\n"}) + "\n"

        # 步骤4: 多路知识召回
        yield json.dumps({"type": "chunk", "content": "> **步骤 4/6: 多路知识召回**\n"}) + "\n"
        
        import time
        t1 = time.time()
        
        # 获取检索统计信息
        total_chunks = 0
        index_status = []
        for db_name in db_names:
            vs = self.kb_service.vector_stores.get(db_name)
            if vs and hasattr(vs, 'index') and vs.index:
                count = vs.index.ntotal if hasattr(vs.index, 'ntotal') else 0
                total_chunks += count
                index_status.append(f"{db_name}({count})")
            else:
                index_status.append(f"{db_name}(未加载)")
        
        yield json.dumps({"type": "chunk", "content": f">   📊 索引状态: {' | '.join(index_status)}\n"}) + "\n"
        yield json.dumps({"type": "chunk", "content": f">   📊 总文档数: {total_chunks} 个片段\n"}) + "\n"
        
        results = self.kb_service.search(search_query, top_k=15, db_names=db_names)
        t2 = time.time()
        
        # 步骤5: 结果融合与重排序
        yield json.dumps({"type": "chunk", "content": "> **步骤 5/6: 结果融合与重排序**\n"}) + "\n"
        
        # 分析检索结果来源
        vector_count = sum(1 for doc, _ in results if doc.metadata.get('db_name') != 'neo4j')
        graph_count = sum(1 for doc, _ in results if doc.metadata.get('db_name') == 'neo4j')
        
        search_details = f"耗时 {round(t2-t1, 2)}s"
        
        yield json.dumps({"type": "chunk", "content": f">   📈 向量检索: 召回 {vector_count} 条\n"}) + "\n"
        yield json.dumps({"type": "chunk", "content": f">   📈 图谱检索: 召回 {graph_count} 条\n"}) + "\n"
        yield json.dumps({"type": "chunk", "content": f">   📈 融合去重: 剩余 {len(results)} 条\n"}) + "\n"
        yield json.dumps({"type": "chunk", "content": f">   ⏱️ 检索耗时: {search_details}\n"}) + "\n"
        
        if results:
            # 重排序分数转换：将原始分数转换为0-1范围便于显示
            def normalize_score(score):
                import math
                return 1 / (1 + math.exp(-score))
            
            yield json.dumps({"type": "chunk", "content": ">   🏆 Top 匹配结果:\n"}) + "\n"
            for i, (doc, score) in enumerate(results[:3], 1):
                match_score = round(normalize_score(score), 3)
                source = doc.metadata.get('doc_name', '未知')
                content_preview = doc.page_content[:50].replace('\n', ' ')
                yield json.dumps({"type": "chunk", "content": f">      #{i} [匹配度:{match_score}] {source}\n"}) + "\n"
                yield json.dumps({"type": "chunk", "content": f">         {content_preview}...\n"}) + "\n"
            
            if len(results) > 3:
                yield json.dumps({"type": "chunk", "content": f">      ... 等共 {len(results)} 条片段\n"}) + "\n"
            
            top_score = float(results[0][1])
        else:
            yield json.dumps({"type": "chunk", "content": ">   ⚠️ 未检索到相关内容\n"}) + "\n"
            top_score = 0.0

        # 步骤6: 质量检查
        yield json.dumps({"type": "chunk", "content": "> **步骤 6/6: 质量检查**\n"}) + "\n"
        
        print(f"[QA Debug] top_score={top_score}, results_count={len(results)}")
        if not enable_tools and len(results) == 0:
            yield json.dumps({"type": "chunk", "content": ">   ❌ 检查未通过: 检索结果为空\n"}) + "\n"
            yield json.dumps({"type": "chunk", "content": "> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"}) + "\n"
            yield json.dumps({"type": "chunk", "content": "抱歉，当前知识库中未找到相关资料，无法提供解答。"}) + "\n"
            yield json.dumps({"type": "sources", "sources": []}) + "\n"
            return

        # 质量兜底检查 - 只在完全为空时拦截，不在分数低时拦截
        if not enable_tools and len(results) == 0:
            yield json.dumps({"type": "chunk", "content": ">   ❌ 检查未通过: 未检索到任何相关资料\n"}) + "\n"
            yield json.dumps({"type": "chunk", "content": "> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"}) + "\n"
            yield json.dumps({"type": "chunk", "content": "抱歉，当前知识库中未找到相关资料。建议检查知识库是否已导入相关文档。"}) + "\n"
            yield json.dumps({"type": "sources", "sources": []}) + "\n"
            return
        
        if not enable_tools or len(results) > 0:
            yield json.dumps({"type": "chunk", "content": f">   ✅ 检查通过: 获取 {len(results)} 条有效资料\n"}) + "\n"
        
        yield json.dumps({"type": "chunk", "content": "> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"}) + "\n"
        # ========== 思考过程结束 ==========
        
        # 4. Prepare Context
        context_parts = []
        sources = []
        source_index = 1
        MAX_CONTEXT_LENGTH = 12000  # 字符上限保护
        current_context_length = 0
            
        for doc, score in results:
            # 解决 ChatPromptTemplate 变量缺失报错：转义检索内容中的单大括号
            safe_content = doc.page_content.replace("{", "{{").replace("}", "}}")
            content_to_add = f"资料 [{source_index}]:\n{safe_content}"
            
            # 检查字符上限
            if current_context_length + len(content_to_add) > MAX_CONTEXT_LENGTH:
                context_parts.append("> ⚠️ 上下文长度已达上限，部分资料已省略")
                break
            
            context_parts.append(content_to_add)
            current_context_length += len(content_to_add)
            doc_name = doc.metadata.get("doc_name", "Unknown")
            
            # 增加返回的参考内容长度，原来是 100 字符，现在扩大到 600 字符，方便前端展示更完整的片段
            original_content = doc.page_content
            display_content = original_content[:600] + "..." if len(original_content) > 600 else original_content
            
            sources.append({
                "content": display_content,
                "doc_id": doc.metadata.get("doc_id"),
                "doc_name": doc_name,
                "score": float(score)
            })
            source_index += 1
            
        context = "\n\n".join(context_parts)
        
        # 动态 Prompt 工程
        base_system_prompt = """你是一位严谨的知识库问答助手。
【最高指令】：
1. 你的所有回答**必须完全、仅限于**参考资料中提供的信息。
2. 绝对不允许使用你的预训练知识进行回答，不允许推测、发散或补充资料中没有的操作步骤、数据或概念。
3. 如果你在参考资料中找不到直接回答用户问题的答案，你必须直接且仅回复：“抱歉，当前知识库中未找到相关资料。”
4. 如果你不知道，就说不知道，不要试图编造答案。

【排版要求】：分步骤条理清晰。你必须在回答中准确引用参考资料的编号（如 [1], [2]），引用的编号必须与参考资料中的 `资料 [X]` 严格对应。在生成答案的末尾，请附上你的回答置信度（高/中/低），并标注引用的资料来源。"""

        if top_score < 0.7 and results:
            base_system_prompt += "\n【注意】当前检索到的资料相关性较低，如果资料确实无法直接回答问题，请直接回答“未找到相关资料”。"
        elif top_score > 0.9:
            base_system_prompt += "\n【注意】当前检索到的资料高度相关，请完全限制在提供资料的范围内回答。"

        base_system_prompt += f"\n\n<参考资料>\n{context}\n</参考资料>"
        
        # 强制在用户提问中再次强调规则
        user_query_with_rule = f"用户问题：{search_query}\n\n请仅使用上面提供的参考资料回答，不要使用任何外部知识。如果资料中没有，请回答未找到相关资料。"
        
        # 5. Agent 执行 (ReAct Pattern)
        try:
            from langchain.agents import AgentExecutor
        except ImportError:
            # Older versions might have it in a different path or we can use custom implementation
            AgentExecutor = None

        try:
            from langchain.agents import create_tool_calling_agent
        except ImportError:
            # Fallback for older langchain versions
            try:
                from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
                from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
            except ImportError:
                format_to_openai_tool_messages = None
                OpenAIToolsAgentOutputParser = None
            create_tool_calling_agent = None

        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        full_answer = ""
        if enable_tools and AgentExecutor is not None:
            try:
                yield json.dumps({"type": "chunk", "content": "> - [ ] 正在初始化 Agent 并规划任务...\n"}) + "\n"
                from app.services.agent_tools import get_tools_by_names
                enabled_tool_names = getattr(Config, 'ENABLED_TOOLS', ['execute_shell_command', 'query_api_endpoint'])
                tools = get_tools_by_names(enabled_tool_names)
                
                if tools:
                    # 使用支持 Tool Calling 的 Agent
                    agent_prompt = ChatPromptTemplate.from_messages([
                        ("system", base_system_prompt),
                        MessagesPlaceholder(variable_name="history", optional=True),
                        ("user", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    
                    if create_tool_calling_agent:
                        agent = create_tool_calling_agent(self.llm, tools, agent_prompt)
                    elif format_to_openai_tool_messages and OpenAIToolsAgentOutputParser:
                        # Fallback implementation if create_tool_calling_agent is not available
                        llm_with_tools = self.llm.bind_tools(tools)
                        agent = (
                            {
                                "input": lambda x: x["input"],
                                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                                    x["intermediate_steps"]
                                ),
                                "history": lambda x: x.get("history", []),
                            }
                            | agent_prompt
                            | llm_with_tools
                            | OpenAIToolsAgentOutputParser()
                        )
                    else:
                        raise Exception("No suitable Agent creation method found in current langchain version.")

                    # max_iterations=3 允许 Agent 多次思考与工具调用 (ReAct模式)
                    agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=3, verbose=True, return_intermediate_steps=True)
                    
                    yield json.dumps({"type": "chunk", "content": "> - [x] Agent 启动，开始推理和工具调用链...\n\n---\n\n"}) + "\n"
                    
                    # 传入裁剪后的历史
                    history_messages = history_obj.messages[-6:] if history_obj else []
                    
                    response = agent_executor.invoke({
                        "input": user_query_with_rule,
                        "history": history_messages
                    })
                    
                    # 输出工具调用的自省过程 (CoT)
                    if response.get("intermediate_steps"):
                        for action, observation in response["intermediate_steps"]:
                            yield json.dumps({"type": "chunk", "content": f"\n\n*🔧 Agent 调用了工具 `{action.tool}`*\n*输入参数: {action.tool_input}*\n*执行结果片段: {str(observation)[:100]}...*\n\n"}) + "\n"
                    
                    full_answer = response["output"]
                    yield json.dumps({"type": "chunk", "content": full_answer}) + "\n"
                else:
                    raise Exception("No tools available")
            except Exception as e:
                print(f"Agent Execution Error: {e}")
                enable_tools = False # 降级为普通生成
                yield json.dumps({"type": "chunk", "content": "> - [!] Agent 执行异常，降级为普通知识问答...\n\n---\n\n"}) + "\n"
        
        if not enable_tools or not full_answer:
            yield json.dumps({"type": "chunk", "content": "> - [x] 开始整合知识生成最终分析报告...\n\n---\n\n"}) + "\n"
            try:
                # 重新构建带历史的 prompt
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", base_system_prompt),
                    MessagesPlaceholder(variable_name="history"),
                    ("user", "{question}")
                ])
                chain = qa_prompt | self.llm
                from langchain_core.runnables.history import RunnableWithMessageHistory
                chain_with_history = RunnableWithMessageHistory(
                    chain,
                    self.get_session_history,
                    input_messages_key="question",
                    history_messages_key="history",
                )
                
                for chunk in chain_with_history.stream(
                    {"question": user_query_with_rule},
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
            
        # 发送引用来源
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        
        # 5. Save to Cache
        result = {
            "answer": full_answer,
            "sources": sources
        }
        if self.use_redis and full_answer and len(full_answer) > 50 and "未找到相关" not in full_answer:
            try:
                self.redis_client.setex(cache_key, timedelta(seconds=3600), json.dumps(result))
            except Exception as e:
                print(f"Redis set error: {e}")
        
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
            print(f"DB Log error: {e}")

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
