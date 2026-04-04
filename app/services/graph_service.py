import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import Config
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

class GraphService:
    def __init__(self):
        self.driver = None
        if GraphDatabase and Config.NEO4J_URI:
            try:
                self.driver = GraphDatabase.driver(
                    Config.NEO4J_URI,
                    auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
                )
                print("Neo4j driver initialized successfully in GraphService.")
            except Exception as e:
                print(f"Failed to initialize Neo4j driver in GraphService: {e}")
                
        # 初始化用于图谱抽取的 LLM
        self.llm = self._init_llm()

    def _init_llm(self):
        active_llm = Config.ACTIVE_LLM.lower()
        if active_llm == 'deepseek':
            return ChatOpenAI(
                base_url=Config.DEEPSEEK_API_URL.replace('/chat/completions', ''), 
                api_key=Config.DEEPSEEK_API_KEY,
                model=Config.DEEPSEEK_MODEL_NAME,
                temperature=0.1
            )
        elif active_llm == 'chatgpt':
            return ChatOpenAI(
                api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
                model='gpt-3.5-turbo',
                temperature=0.1
            )
        elif active_llm == 'vllm':
            base_url = Config.VLLM_API_URL.replace('/chat/completions', '') if Config.VLLM_API_URL else "http://127.0.0.1:8000/v1"
            return ChatOpenAI(
                base_url=base_url, 
                api_key=Config.VLLM_API_KEY,
                model=Config.VLLM_MODEL_NAME,
                temperature=0.1
            )
        else:
            return ChatOpenAI(
                base_url=Config.QWEN_API_URL.replace('/chat/completions', '') if hasattr(Config, 'QWEN_API_URL') else "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                api_key=os.getenv('DASHSCOPE_API_KEY', getattr(Config, 'QWEN_API_KEY', 'EMPTY')),
                model="qwen-plus",
                temperature=0.1
            )

    def extract_and_store_graph(self, chunks, doc_id):
        """利用 LLM 从文本块中抽取实体和关系并存入 Neo4j (Graph RAG 构建)"""
        if not self.driver:
            # print("Graph RAG: Neo4j not configured, skipping graph extraction.") # 注释掉避免用户误认为是bug
            return

        prompt = ChatPromptTemplate.from_messages([
            ("system", '''你是一个信息抽取专家。请从给定的文本中提取关键实体和它们之间的关系。
实体类型包括但不限于：
- 技术/模型：算法、模型、框架、工具（如 YOLO, SSD, BERT, TensorFlow）
- 基础设施：服务器、服务、配置、数据库（如 Nginx, MySQL, Redis）
- 概念：技术概念、方法、流程（如 目标检测, 特征融合, 量化部署）

关系类型包括但不限于：
- EVOLVED_FROM/EVOLVED_TO：版本演进（如 YOLOv2 → YOLOv3）
- DEPENDS_ON：依赖关系
- HAS_COMPONENT：组成关系（如 YOLOv3 包含 Darknet53）
- COMPARED_WITH：对比关系
- CONFIGURED_BY/HAS_ERROR/RELATED_TO 等

请返回严格的 JSON 格式，包含 entities 和 relations 两个列表。
实体格式: {{"id": "唯一标识", "type": "实体类型", "name": "名称"}}
关系格式: {{"source": "源实体id", "target": "目标实体id", "type": "关系类型", "source_type": "源实体类型", "target_type": "目标实体类型"}}
特别注意：当文本提及模型/算法的版本演进时，务必提取 EVOLVED_FROM/EVOLVED_TO 关系。
如果未发现明显关系，返回空列表。请只输出 JSON 字符串，不要输出 Markdown 格式标记。'''),
            ("user", "文本内容：\n{text}")
        ])
        
        chain = prompt | self.llm
        
        # 完整实现：处理所有文本块
        with self.driver.session() as session:
            for i, chunk in enumerate(chunks):
                text = chunk.content
                # 过滤掉过短的无意义文本块
                if len(text.strip()) < 20:
                    continue
                    
                try:
                    # 适当限制单次请求文本长度，避免超出上下文限制
                    response = chain.invoke({"text": text[:2000]}) 
                    result_text = response.content.strip()
                    # 清理可能的 markdown 标记
                    if result_text.startswith("```json"):
                        result_text = result_text[7:-3].strip()
                    elif result_text.startswith("```"):
                        result_text = result_text[3:-3].strip()
                        
                    # 尝试解析 JSON
                    import re
                    json_match = re.search(r'(\{.*\})', result_text, re.DOTALL)
                    if json_match:
                        result_text = json_match.group(1)
                        
                    data = json.loads(result_text)
                    entities = data.get("entities", [])
                    relations = data.get("relations", [])
                    
                    # 存入 Neo4j
                    for entity in entities:
                        # 构建全局唯一ID：仅基于名称，避免因类型不一致导致图分裂
                        entity_name = str(entity.get('name', '')).strip()
                        if not entity_name:
                            continue
                        entity_id = f"entity_{entity_name}".lower()
                        session.run(
                            """MERGE (e:Entity {id: $id}) 
                            ON CREATE SET e.name = $name, e.type = $type
                            WITH e
                            SET e.doc_ids = CASE 
                                WHEN $doc_id IN coalesce(e.doc_ids, []) THEN e.doc_ids 
                                ELSE coalesce(e.doc_ids, []) + [$doc_id] 
                            END""",
                            id=entity_id, name=entity_name, type=entity.get('type', 'Unknown'), doc_id=doc_id
                        )
                        
                    for rel in relations:
                        source_name = str(rel.get('source', '')).strip()
                        target_name = str(rel.get('target', '')).strip()
                        if not source_name or not target_name:
                            continue
                        # 构建全局唯一ID
                        source_id = f"entity_{source_name}".lower()
                        target_id = f"entity_{target_name}".lower()
                        rel_type = str(rel.get('type', 'RELATED_TO')).replace(' ', '_').upper()
                        # 确保关系类型是合法的Neo4j类型
                        import re
                        rel_type = re.sub(r'[^A-Z0-9_]', '_', rel_type)
                        if not rel_type:
                            rel_type = 'RELATED_TO'
                            
                        confidence = rel.get('confidence', 0.8)
                        chunk_id = getattr(chunk, 'chunk_id', None)
                        # 这里如果节点不存在，先MERGE节点，防止关系孤立丢失
                        session.run(
                            f"""
                            MERGE (a:Entity {{id: $source}})
                            ON CREATE SET a.name = $source_name, a.type = $source_type
                            MERGE (b:Entity {{id: $target}})
                            ON CREATE SET b.name = $target_name, b.type = $target_type
                            WITH a, b
                            MERGE (a)-[r:{rel_type}]->(b)
                            SET r.confidence = $confidence, 
                                r.source_doc_id = $doc_id,
                                r.source_chunk_id = $chunk_id,
                                r.updated_at = datetime()""",
                            source=source_id, target=target_id, 
                            source_name=source_name, source_type=rel.get('source_type', 'Unknown'),
                            target_name=target_name, target_type=rel.get('target_type', 'Unknown'),
                            confidence=confidence, doc_id=doc_id, chunk_id=chunk_id
                        )
                except Exception as e:
                    print(f"Graph extraction error on chunk {i}: {e}")

    def extract_entities_from_query(self, query):
        """利用 LLM 动态提取查询中的实体，增强图谱检索召回率"""
        if not self.llm:
            return []
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个命名实体识别专家。请从用户的查询中提取出核心运维实体名称（如具体的服务器名称、IP、服务名如Zabbix、错误码等）。只返回实体名称列表，以逗号分隔，不要返回其他任何内容。如果没有实体，返回空。"),
            ("user", "{query}")
        ])
        try:
            response = (prompt | self.llm).invoke({"query": query})
            entities = [e.strip() for e in response.content.split(',') if e.strip()]
            return [e for e in entities if e]
        except Exception as e:
            print(f"LLM entity extraction failed: {e}")
            return []
