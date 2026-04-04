import os
import sys
import re
import numpy as np

# 将项目根目录添加到 sys.path，以便能够正确导入 app 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangchainDocument
from app.config import Config
from app.db import db
from app.db.models import Chunk, VectorIndex, Document
from datetime import datetime
from rank_bm25 import BM25Okapi
from functools import lru_cache
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

# 尝试导入jieba进行中文分词
try:
    import jieba
    # 延迟初始化，避免启动时加载过慢
    JIEBA_AVAILABLE = True
except ImportError as e:
    JIEBA_AVAILABLE = False
    print(f"⚠ Warning: jieba not available ({e}), falling back to tokenizer-based segmentation")

class KnowledgeBaseService:
    def __init__(self):
        # 从统一配置中读取模型路径
        self.embedding_model_name = Config.EMBEDDING_MODEL_PATH
        self.index_path = Config.FAISS_INDEX_PATH
        # 使用本地路径加载 HuggingFaceEmbeddings，并启用 GPU 加速
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True} # BGE 模型推荐开启归一化
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # 引入重排序模型，并将其放置到 GPU 上以提升推理速度
        self.reranker_model_name = Config.RERANKER_MODEL_PATH
        print("Loading reranker model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name)
        self.reranker_model.to(self.device)
        self.reranker_model.eval()
        
        # 打印GPU设备信息
        print(f"[KnowledgeBase] Embedding device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print(f"[KnowledgeBase] Reranker device: {self.device}")
        
        self.vector_store = None
        self.bm25 = None
        self.chunk_map = {} # Map chunk_id to Chunk object for BM25
        
        # 为了支持多库，我们需要维护多个 FAISS 索引实例和 BM25 实例
        # 结构: {"db_name": vector_store}
        self.vector_stores = {}
        # 结构: {"db_name": {"bm25": bm25_instance, "chunk_map": {}}}
        self.bm25_stores = {}
        
        # 查询向量缓存，用于语义去重
        self._query_embedding_cache = {}
        
        # 索引统计信息缓存
        self._index_stats = {}
        
        # Initialize Neo4j Driver for Graph RAG if configured
        self.neo4j_driver = None
        if GraphDatabase and Config.NEO4J_URI:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    Config.NEO4J_URI, 
                    auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
                    connection_timeout=30,
                    max_transaction_retry_time=30
                )
                print("Neo4j driver initialized successfully for Graph RAG.")
            except Exception as e:
                print(f"Failed to initialize Neo4j driver: {e}")
        
        self.load_all_indexes()

    def get_index_path(self, db_name):
        """获取指定数据库的索引保存路径"""
        db_index_path = os.path.join(self.index_path, db_name)
        if not os.path.exists(db_index_path):
            os.makedirs(db_index_path, exist_ok=True)
        return db_index_path

    def load_all_indexes(self):
        """Load all FAISS indexes from disk if exists."""
        # 先加载默认库
        self.load_index('default')
        
        # 尝试加载其他库
        if os.path.exists(self.index_path):
            for db_name in os.listdir(self.index_path):
                if os.path.isdir(os.path.join(self.index_path, db_name)) and db_name != 'default':
                    self.load_index(db_name)

    def load_index(self, db_name='default'):
        """Load specific FAISS index from disk if exists."""
        db_index_path = self.get_index_path(db_name)
        # FAISS 要求目录下必须有 index.faiss 文件才能加载
        if os.path.exists(os.path.join(db_index_path, "index.faiss")):
            try:
                vs = FAISS.load_local(
                    db_index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                self.vector_stores[db_name] = vs
                print(f"Loaded FAISS index for '{db_name}' from {db_index_path}")
            except Exception as e:
                print(f"Failed to load index for '{db_name}': {e}")
        else:
            # 不要抛出异常或中断，只是打印一条提示即可，新部署或清空后这是正常现象
            # print(f"No existing FAISS index found for '{db_name}'.") # 注释掉避免用户误认为是bug
            self.vector_stores[db_name] = None
            
        self.build_bm25_index(db_name)

    def _tokenize_for_bm25(self, text):
        """
        BM25分词优化：优先使用jieba中文分词，回退到tokenizer
        使用完整内容（最多4000字符）构建索引，避免关键词遗漏
        """
        # 限制最大长度避免内存问题，但比之前的800字符更宽松
        max_length = 4000
        truncated_text = text[:max_length] if len(text) > max_length else text
        
        if JIEBA_AVAILABLE:
            # 使用jieba进行中文分词，效果更好
            tokens = list(jieba.cut(truncated_text))
            # 过滤空字符串和标点
            tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]
            return tokens
        else:
            # 回退到tokenizer
            return self.reranker_tokenizer.tokenize(truncated_text)
    
    def build_bm25_index(self, db_name='default'):
        """Build BM25 index from DB chunks for specific db_name."""
        from app.db.models import Document, Chunk
        from flask import has_app_context
        
        # 确保在应用上下文中执行数据库查询
        if not has_app_context():
            from app.main import app
            with app.app_context():
                return self._build_bm25_index_impl(db_name)
        return self._build_bm25_index_impl(db_name)
    
    def _build_bm25_index_impl(self, db_name='default'):
        """BM25索引构建的实际实现"""
        from app.db.models import Document, Chunk
        try:
            chunks = Chunk.query.join(Document).filter(Document.db_name == db_name).all()
            if not chunks:
                self.bm25_stores[db_name] = {"bm25": None, "chunk_map": {}, "chunk_list": []}
                return
                
            tokenized_corpus = []
            chunk_map = {}
            chunk_list = []  # 保持顺序的列表
            
            for chunk in chunks:
                # 使用优化的分词方法（jieba中文分词），使用完整内容构建索引
                tokens = self._tokenize_for_bm25(chunk.content)
                tokenized_corpus.append(tokens)
                chunk_map[chunk.chunk_id] = chunk
                chunk_list.append(chunk)  # 保持与tokenized_corpus相同的顺序
                
            bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_stores[db_name] = {"bm25": bm25, "chunk_map": chunk_map, "chunk_list": chunk_list}
            print(f"Built BM25 index for '{db_name}' with {len(chunks)} chunks.")
        except Exception as e:
            print(f"Error building BM25 index for {db_name}: {e}")
            import traceback
            traceback.print_exc()
            self.bm25_stores[db_name] = {"bm25": None, "chunk_map": {}, "chunk_list": []}

    def remove_from_bm25_index(self, doc_id, db_name='default'):
        """
        从BM25索引中移除指定文档的chunks，用于增量更新。
        删除文档后调用此方法重建该知识库的BM25索引。
        
        Args:
            doc_id: 要移除的文档ID
            db_name: 知识库名称
        """
        from flask import has_app_context
        
        # 确保在应用上下文中执行数据库查询
        if not has_app_context():
            from app.main import app
            with app.app_context():
                return self._remove_from_bm25_index_impl(doc_id, db_name)
        return self._remove_from_bm25_index_impl(doc_id, db_name)
    
    def _remove_from_bm25_index_impl(self, doc_id, db_name='default'):
        """从BM25索引中移除文档的实际实现"""
        from app.db.models import Document, Chunk
        try:
            # 获取该知识库下当前所有有效的chunks（排除被删除文档的chunks）
            chunks = Chunk.query.join(Document).filter(
                Document.db_name == db_name,
                Chunk.doc_id != doc_id
            ).all()
            
            if not chunks:
                # 如果没有chunks了，清空BM25索引
                self.bm25_stores[db_name] = {"bm25": None, "chunk_map": {}, "chunk_list": []}
                print(f"Cleared BM25 index for '{db_name}' after removing doc {doc_id}")
                return
                
            tokenized_corpus = []
            chunk_map = {}
            chunk_list = []
            
            for chunk in chunks:
                # 使用优化的分词方法（jieba中文分词），使用完整内容构建索引
                tokens = self._tokenize_for_bm25(chunk.content)
                tokenized_corpus.append(tokens)
                chunk_map[chunk.chunk_id] = chunk
                chunk_list.append(chunk)
                
            bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_stores[db_name] = {"bm25": bm25, "chunk_map": chunk_map, "chunk_list": chunk_list}
            print(f"Rebuilt BM25 index for '{db_name}' after removing doc {doc_id}, remaining {len(chunks)} chunks.")
        except Exception as e:
            print(f"Error rebuilding BM25 index for {db_name} after removing doc {doc_id}: {e}")
            import traceback
            traceback.print_exc()

    def build_index(self, db_name='default'):
        """Build index from all chunks in specific DB."""
        from flask import has_app_context
        
        # 确保在应用上下文中执行数据库查询
        if not has_app_context():
            from app.main import app
            with app.app_context():
                return self._build_index_impl(db_name)
        return self._build_index_impl(db_name)
    
    def _build_index_impl(self, db_name='default'):
        """构建索引的实际实现"""
        from app.db.models import Document, Chunk
        chunks = Chunk.query.join(Document).filter(Document.db_name == db_name).all()
        if not chunks:
            print(f"No chunks to index for '{db_name}'.")
            # 如果该库被清空了，清理内存中的实例并删除磁盘上的旧索引文件
            if db_name in self.vector_stores:
                del self.vector_stores[db_name]
            self.bm25_stores[db_name] = {"bm25": None, "chunk_map": {}, "chunk_list": []}
            db_index_path = self.get_index_path(db_name)
            faiss_file = os.path.join(db_index_path, "index.faiss")
            pkl_file = os.path.join(db_index_path, "index.pkl")
            if os.path.exists(faiss_file):
                os.remove(faiss_file)
            if os.path.exists(pkl_file):
                os.remove(pkl_file)
            return

        from app.db import db
        from app.db.models import Document
        
        # 预先获取文档映射，避免在循环中触发 N+1 查询或 DetachedInstanceError
        doc_ids = list(set([chunk.doc_id for chunk in chunks if chunk.doc_id]))
        doc_map = {}
        try:
            docs = db.session.query(Document).filter(Document.doc_id.in_(doc_ids)).all()
            doc_map = {doc.doc_id: doc.doc_name for doc in docs}
        except Exception as e:
            print(f"Warning: Failed to fetch document map for indexing: {e}")

        documents = []
        for chunk in chunks:
            doc_name = doc_map.get(chunk.doc_id, "Unknown")

            meta = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_name": doc_name,
                "chunk_index": chunk.chunk_index,
                "db_name": db_name
            }
            doc = LangchainDocument(page_content=chunk.content, metadata=meta)
            documents.append(doc)

        print(f"Creating index for '{db_name}' with {len(documents)} documents...")
        
        # FAISS自适应索引优化：根据数据量选择最优索引类型
        vs = self._create_adaptive_faiss_index(documents, db_name)
        self.vector_stores[db_name] = vs
        
        # 更新索引统计信息
        self._index_stats[db_name] = {
            'vector_count': len(documents),
            'dimension': len(self.embeddings.embed_query("test")),
            'index_type': self._get_index_type_name(len(documents))
        }
        
        db_index_path = self.get_index_path(db_name)
        
        # FAISS GPU索引需转回CPU才能保存
        try:
            import faiss
            if hasattr(faiss, 'index_gpu_to_cpu') and hasattr(vs.index, 'getDevice'):
                try:
                    vs.index = faiss.index_gpu_to_cpu(vs.index)
                except:
                    pass  # 已在CPU上，忽略
        except Exception:
            pass
        
        vs.save_local(db_index_path)
        print(f"Index saved to {db_index_path}")
        
        # 兼容旧代码，将默认库的指标记录下来
        if db_name == 'default':
            new_index = VectorIndex(chunk_count=len(documents))
            db.session.add(new_index)
            db.session.commit()
        
        self.build_bm25_index(db_name)

    def _get_index_type_name(self, num_vectors):
        """根据向量数量返回索引类型名称"""
        if num_vectors < 1000:
            return "IndexFlatIP"
        elif num_vectors < 10000:
            return "IndexIVFFlat"
        else:
            return "IndexIVFPQ"
    
    def _create_adaptive_faiss_index(self, documents, db_name='default'):
        """
        FAISS自适应索引优化：根据数据量自动选择最优索引类型
        - < 1000条: IndexFlatIP (精确搜索)
        - 1000-10000条: IndexIVFFlat (近似搜索，平衡速度与精度)
        - > 10000条: IndexIVFPQ (乘积量化，大幅降低内存)
        """
        import faiss
        
        # 检测FAISS GPU支持
        try:
            faiss_gpu_available = hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
        except:
            faiss_gpu_available = False
        
        num_docs = len(documents)
        print(f"Building adaptive FAISS index for {num_docs} documents...")
        
        # 获取向量维度
        sample_embedding = self.embeddings.embed_query("sample text")
        dimension = len(sample_embedding)
        
        if num_docs < 1000:
            # 小数据量：使用精确搜索，内存友好
            print(f"Using IndexFlatIP (exact search) for small dataset")
            vs = FAISS.from_documents(documents, self.embeddings)
            
        elif num_docs < 10000:
            # 中等数据量：使用IVF近似搜索
            print(f"Using IndexIVFFlat (approximate search) for medium dataset")
            # nlist设置为sqrt(num_docs)*2，上限256，在搜索精度和速度间取得平衡
            nlist = min(int(np.sqrt(num_docs)) * 2, 256)
            nlist = max(nlist, 1)  # 至少1个中心
            
            # 先创建基础索引
            vs = FAISS.from_documents(documents, self.embeddings)
            
            # 转换为IVF索引
            base_index = vs.index
            quantizer = faiss.IndexFlatIP(dimension)
            ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # 训练并添加向量
            embeddings = np.array([self.embeddings.embed_query(doc.page_content) 
                                  for doc in documents]).astype('float32')
            ivf_index.train(embeddings)
            ivf_index.add(embeddings)
            
            # 替换索引
            vs.index = ivf_index
            
            # 尝试将索引转移到GPU
            if faiss_gpu_available:
                try:
                    res = faiss.StandardGpuResources()
                    vs.index = faiss.index_cpu_to_gpu(res, 0, vs.index)
                    print(f"FAISS index transferred to GPU")
                except Exception as e:
                    print(f"FAISS GPU transfer failed, keeping CPU: {e}")
            
        else:
            # 大数据量：使用PQ乘积量化
            print(f"Using IndexIVFPQ (product quantization) for large dataset")
            # nlist设置为sqrt(num_docs)*2，上限256，提供更多聚类中心以提高精度
            nlist = min(int(np.sqrt(num_docs)) * 2, 256)
            nlist = max(nlist, 1)
            # m值根据维度动态计算，确保m能整除dimension，每个子向量约8维
            m = min(64, dimension // 8)
            # 确保m能整除dimension
            while dimension % m != 0 and m > 1:
                m -= 1
            
            vs = FAISS.from_documents(documents, self.embeddings)
            
            # 转换为IVFPQ索引
            base_index = vs.index
            quantizer = faiss.IndexFlatIP(dimension)
            ivfpq_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            
            embeddings = np.array([self.embeddings.embed_query(doc.page_content) 
                                  for doc in documents]).astype('float32')
            ivfpq_index.train(embeddings)
            ivfpq_index.add(embeddings)
            
            vs.index = ivfpq_index
            
            # 尝试将索引转移到GPU
            if faiss_gpu_available:
                try:
                    res = faiss.StandardGpuResources()
                    vs.index = faiss.index_cpu_to_gpu(res, 0, vs.index)
                    print(f"FAISS index transferred to GPU")
                except Exception as e:
                    print(f"FAISS GPU transfer failed, keeping CPU: {e}")
        
        return vs
    
    def add_documents(self, chunks, db_name='default'):
        """Incremental update (add new chunks to specific db)."""
        vs = self.vector_stores.get(db_name)
        if not vs:
            # 如果索引不存在，先构建全量索引（包含新文档）
            print(f"[Add Documents] No existing index for '{db_name}', building full index...")
            self.build_index(db_name)
            return
        
        if not chunks:
            print(f"[Add Documents] No chunks to add for '{db_name}'")
            return

        from app.db import db
        from app.db.models import Document
        
        # 预先获取文档映射，避免在循环中触发 N+1 查询或 DetachedInstanceError
        doc_ids = list(set([chunk.doc_id for chunk in chunks if chunk.doc_id]))
        doc_map = {}
        try:
            docs = db.session.query(Document).filter(Document.doc_id.in_(doc_ids)).all()
            doc_map = {doc.doc_id: doc.doc_name for doc in docs}
        except Exception as e:
            print(f"Warning: Failed to fetch document map for adding documents: {e}")

        documents = []
        for chunk in chunks:
            doc_name = doc_map.get(chunk.doc_id, "Unknown")
                    
            meta = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_name": doc_name,
                "chunk_index": chunk.chunk_index,
                "db_name": db_name
            }
            doc = LangchainDocument(page_content=chunk.content, metadata=meta)
            documents.append(doc)
            
        vs.add_documents(documents)
        db_index_path = self.get_index_path(db_name)
        vs.save_local(db_index_path)
        print(f"Added {len(documents)} documents to index '{db_name}'.")
        
        self.build_bm25_index(db_name)

    def _extract_entities_from_query(self, query):
        """
        利用词典/NER提取查询中的实体，替代大模型抽取以降低延迟
        支持技术术语、版本号、关系链提取
        """
        import re
        entities = []
        
        # 1. 预置运维词库
        ops_dict = ["服务器", "数据库", "Zabbix", "Nginx", "MySQL", "网络", "Redis", "宕机", 
                   "CPU", "内存", "磁盘", "Tomcat", "微服务", "网关", "Docker", "Kubernetes",
                   "Linux", "Windows", "Apache", "Elasticsearch", "Kafka", "RabbitMQ"]
        
        # 2. 匹配运维词典
        for kw in ops_dict:
            if kw.lower() in query.lower():
                entities.append(kw)
        
        # 3. 提取技术术语（如 YOLOv2, BERT-base, GPT-4, SSD 等）
        # 匹配模式：字母+数字/点/下划线组合（不区分大小写）
        tech_patterns = [
            r'[A-Za-z][a-zA-Z]*\d+(?:\.\d+)?[a-z]*',  # YOLOv2, yolo3, BERT4
            r'[A-Za-z][a-zA-Z]+[-_][A-Za-z0-9]+',       # BERT-base, GPT-4
            r'[a-z]+[A-Z][a-zA-Z0-9]*',                  # camelCase技术词
            r'[A-Z]{2,}(?:v\d+)?',                       # SSD, YOLO, CNN, RNN, YOLOv5
        ]
        for pattern in tech_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # 3.1 大小写归一化：将提取的技术术语统一为常见形式
        normalized = []
        for e in entities:
            # 保留原始形式
            normalized.append(e)
            # 同时添加大写版本以匹配图谱中的存储形式
            if e != e.upper() and len(e) <= 10:
                normalized.append(e.upper())
        entities = normalized
        
        # 4. 提取中文实体词（2-10个中文字符的连续词组）
        chinese_entities = re.findall(r'[\u4e00-\u9fa5]{2,10}', query)
        # 过滤常见停用词
        stop_words = {'什么', '怎么', '如何', '为什么', '多少', '哪些', '这个', '那个', '进行', '使用'}
        chinese_entities = [e for e in chinese_entities if e not in stop_words and len(e) >= 2]
        entities.extend(chinese_entities)
        
        # 5. 提取关系链指示词（如 A与B的区别、A和B的关系、A与B的关联）
        relation_patterns = [
            r'(\w+)与(\w+)的?(?:区别|关系|关联|对比|联系)',
            r'(\w+)和(\w+)的?(?:区别|关系|关联|对比|联系)',
            r'(\w+)到(\w+)的?路径',
            r'(\w+)跟(\w+)的?(?:区别|关系|关联)',
        ]
        for pattern in relation_patterns:
            match = re.search(pattern, query)
            if match:
                entities.extend([match.group(1), match.group(2)])
        
        # 去重并过滤
        unique_entities = list(set([e.strip() for e in entities if len(e.strip()) >= 2]))
        print(f"[Entity Extraction] Query: '{query}' -> Entities: {unique_entities}")
        return unique_entities

    def graph_search(self, query):
        """
        执行图数据库检索 (Graph RAG)，包含动态深度与个性化排序
        支持关系链追溯：a->b->c->d
        """
        if not self.neo4j_driver:
            print("[Graph Search] Neo4j driver not available")
            return []
            
        entities = self._extract_entities_from_query(query)
        if not entities:
            print("[Graph Search] No entities extracted from query")
            return []
            
        graph_results = []
        seen_relations = set()  # 去重用
        
        try:
            with self.neo4j_driver.session() as session:
                # 1. 单实体查询（多跳关系追溯）
                for entity in entities:
                    print(f"[Graph Search] Searching for entity: {entity}")
                    
                    # 1.1 直接匹配节点名称（精确匹配优先）
                    cypher_exact = """
                    MATCH (n)
                    WHERE n.name = $entity OR n.id = $entity
                    RETURN n.name as name, n.id as id, labels(n) as labels
                    LIMIT 5
                    """
                    result = session.run(cypher_exact, entity=entity)
                    exact_nodes = list(result)
                    print(f"[Graph Search] Exact match nodes: {len(exact_nodes)}")
                    
                    # 1.2 1跳关系查询
                    cypher_1_hop = """
                    MATCH (n)-[r]-(m)
                    WHERE (n.name = $entity OR n.id = $entity OR n.name CONTAINS $entity)
                      AND n.name <> m.name
                    RETURN n.name as source, type(r) as relation, m.name as target, 
                           m.description as target_desc, r.properties as rel_props
                    LIMIT 20
                    """
                    try:
                        result = session.run(cypher_1_hop, entity=entity)
                        records_1hop = list(result)
                    except Exception as e:
                        logging.warning(f"1-hop查询失败: {e}")
                        records_1hop = []
                    print(f"[Graph Search] 1-hop relations: {len(records_1hop)}")
                    
                    # 1.3 始终执行多跳查询（关系链追溯 a->b->c->d）
                    records_2hop = []
                    records_3hop = []
                    
                    # 2-hop查询
                    cypher_2_hop = """
                    MATCH path = (n)-[r1]-(m1)-[r2]-(m2)
                    WHERE (n.name = $entity OR n.id = $entity OR n.name CONTAINS $entity)
                      AND n.name <> m1.name AND m1.name <> m2.name AND n.name <> m2.name
                    WITH n, m1, m2, r1, r2, path
                    RETURN n.name as source, 
                           type(r1) as relation1, m1.name as mid,
                           type(r2) as relation2, m2.name as target,
                           length(path) as path_len
                    LIMIT 15
                    """
                    try:
                        result = session.run(cypher_2_hop, entity=entity)
                        records_2hop = list(result)
                    except Exception as e:
                        logging.warning(f"2-hop查询失败: {e}")
                        records_2hop = []
                    print(f"[Graph Search] 2-hop relations: {len(records_2hop)}")
                    
                    # 3-hop查询（支持 a->b->c->d 关系链追溯）
                    cypher_3_hop = """
                    MATCH path = (n)-[r1]-(m1)-[r2]-(m2)-[r3]-(m3)
                    WHERE (n.name = $entity OR n.id = $entity OR n.name CONTAINS $entity)
                      AND n.name <> m1.name AND m1.name <> m2.name AND m2.name <> m3.name
                      AND n.name <> m2.name AND n.name <> m3.name AND m1.name <> m3.name
                    RETURN n.name as source, type(r1) as rel1, m1.name as mid1,
                           type(r2) as rel2, m2.name as mid2,
                           type(r3) as rel3, m3.name as target,
                           n.type as source_type, m3.type as target_type
                    LIMIT 10
                    """
                    try:
                        result = session.run(cypher_3_hop, entity=entity)
                        records_3hop = list(result)
                    except Exception as e:
                        logging.warning(f"3-hop查询失败: {e}")
                        records_3hop = []
                    print(f"[Graph Search] 3-hop relations: {len(records_3hop)}")
                    
                    # 处理1跳结果
                    for record in records_1hop:
                        rel_key = f"{record['source']}-{record['relation']}-{record['target']}"
                        if rel_key not in seen_relations:
                            seen_relations.add(rel_key)
                            target_desc = record.get('target_desc', '')
                            desc_text = f" ({target_desc})" if target_desc else ""
                            relation_desc = f"【图谱知识】{record['source']} → [{record['relation']}] → {record['target']}{desc_text}"
                            
                            doc = LangchainDocument(
                                page_content=relation_desc,
                                metadata={
                                    "doc_id": f"graph_{record['target']}",
                                    "doc_name": "知识图谱",
                                    "db_name": "neo4j",
                                    "chunk_id": f"graph_{record['target']}",
                                    "source_entity": record['source'],
                                    "relation": record['relation'],
                                    "target_entity": record['target']
                                }
                            )
                            graph_results.append((doc, 0.92))
                    
                    # 处理2跳结果（关系链）- 置信度：0.92 * 0.85 = 0.78
                    for record in records_2hop:
                        rel_key = f"{record['source']}-{record['relation1']}-{record['mid']}-{record['relation2']}-{record['target']}"
                        if rel_key not in seen_relations:
                            seen_relations.add(rel_key)
                            relation_desc = f"【关系链】{record['source']} → [{record['relation1']}] → {record['mid']} → [{record['relation2']}] → {record['target']}"
                            
                            doc = LangchainDocument(
                                page_content=relation_desc,
                                metadata={
                                    "doc_id": f"graph_chain_{record['target']}",
                                    "doc_name": "知识图谱-关系链",
                                    "db_name": "neo4j",
                                    "chunk_id": f"graph_chain_{record['target']}",
                                    "path_length": 2,
                                    "is_relation_chain": True
                                }
                            )
                            graph_results.append((doc, 0.78))
                    
                    # 处理3跳结果（关系链）- 置信度：0.92 * 0.85 * 0.85 = 0.66
                    for record in records_3hop:
                        rel_key = f"{record['source']}-{record['rel1']}-{record['mid1']}-{record['rel2']}-{record['mid2']}-{record['rel3']}-{record['target']}"
                        if rel_key not in seen_relations:
                            seen_relations.add(rel_key)
                            relation_desc = f"【关系链】{record['source']} → [{record['rel1']}] → {record['mid1']} → [{record['rel2']}] → {record['mid2']} → [{record['rel3']}] → {record['target']}"
                            
                            doc = LangchainDocument(
                                page_content=relation_desc,
                                metadata={
                                    "doc_id": f"graph_chain3_{record['target']}",
                                    "doc_name": "知识图谱-关系链(3-hop)",
                                    "db_name": "neo4j",
                                    "chunk_id": f"graph_chain3_{record['target']}",
                                    "path_length": 3,
                                    "is_relation_chain": True
                                }
                            )
                            graph_results.append((doc, 0.66))
                
                # 2. 多实体联合查询（如 "YOLOv2与YOLOv3的区别"）
                if len(entities) >= 2:
                    print(f"[Graph Search] Multi-entity query: {entities}")
                    cypher_multi = """
                    MATCH path = (n1)-[r*1..6]-(n2)
                    WHERE any(e IN $entities WHERE n1.name CONTAINS e OR n1.name = e OR n1.id = e)
                      AND any(e IN $entities WHERE n2.name CONTAINS e OR n2.name = e OR n2.id = e)
                      AND n1 <> n2
                    RETURN n1.name as source, n2.name as target,
                           [rel in r | type(rel)] as relations,
                           [node in nodes(path) | node.name] as path_nodes,
                           length(path) as path_len
                    ORDER BY length(path) ASC
                    LIMIT 10
                    """
                    result = session.run(cypher_multi, entities=entities)
                    multi_records = list(result)
                    print(f"[Graph Search] Multi-entity paths: {len(multi_records)}")
                    
                    for record in multi_records:
                        rels = " → ".join(record['relations'])
                        path_nodes = record.get('path_nodes', [])
                        path_len = record.get('path_len', 1)
                        if path_nodes:
                            path_str = " → ".join([str(n) for n in path_nodes if n])
                            relation_desc = f"【实体关联】{record['source']} 与 {record['target']} 的关系链({path_len}跳): {path_str}（关系类型: {rels}）"
                        else:
                            relation_desc = f"【实体关联】{record['source']} 与 {record['target']} 通过路径关联: {rels}"
                        
                        # 置信度随跳数衰减
                        confidence = max(0.5, 0.92 * (0.85 ** (path_len - 1)))
                        
                        doc = LangchainDocument(
                            page_content=relation_desc,
                            metadata={
                                "doc_id": f"graph_multi_{record['source']}_{record['target']}",
                                "doc_name": "知识图谱-实体关联",
                                "db_name": "neo4j",
                                "chunk_id": f"graph_multi_{record['source']}",
                                "is_multi_entity": True
                            }
                        )
                        graph_results.append((doc, confidence))
                        
        except Exception as e:
            print(f"[Graph Search Error] {e}")
            import traceback
            traceback.print_exc()
            
        print(f"[Graph Search] Total results: {len(graph_results)}")
        return graph_results

    def _analyze_query_intent(self, query):
        """
        查询意图分析：识别查询类型，动态调整检索策略
        使用评分制，所有规则都参与评估，取最高分意图
        返回: {
            'is_exact_query': bool,  # 是否精确查询（含代码/缩写）
            'is_troubleshooting': bool,  # 是否故障排查
            'is_comparison': bool,  # 是否对比查询
            'is_factual': bool,  # 是否事实查询
            'weights': {'vector': float, 'bm25': float, 'graph': float}  # 动态权重
        }
        """
        import re
        intent_scores = {}
            
        # 精确查询检测：包含代码、缩写、特殊符号
        has_special_chars = bool(re.search(r'[{}\[\]()<>=;|&]', query))
        if has_special_chars or '代码' in query or 'code' in query.lower():
            intent_scores['precise'] = intent_scores.get('precise', 0) + 2
        # 包含英文数字符号等技术术语也属于精确查询
        if bool(re.search(r'[a-zA-Z0-9_]{3,}', query)) or bool(re.search(r'[\(\)\[\]\{\}@#$%^&*]', query)):
            intent_scores['precise'] = intent_scores.get('precise', 0) + 1
            
        # 故障排查检测
        fault_keywords = ['错误', '失败', '异常', '报错', '故障', '超时', '崩溃', '宕机', 'error', 'fail', 'timeout']
        fault_count = sum(1 for kw in fault_keywords if kw in query.lower())
        if fault_count > 0:
            intent_scores['fault'] = fault_count * 1.5
        # 兼容原有的故障排查关键词（问题、无法、不能语义较弱）
        weak_fault_keywords = ['问题', '无法', '不能']
        weak_fault_count = sum(1 for kw in weak_fault_keywords if kw in query)
        if weak_fault_count > 0:
            intent_scores['fault'] = intent_scores.get('fault', 0) + weak_fault_count * 0.5
            
        # 对比查询检测
        comparison_keywords = ['区别', '对比', '比较', '不同', 'vs', '差异', 'versus']
        comparison_count = sum(1 for kw in comparison_keywords if kw in query.lower())
        if comparison_count > 0:
            intent_scores['compare'] = comparison_count * 2
            
        # 事实查询检测
        factual_keywords = ['是什么', '怎么', '如何', '什么是', '为什么', '原理', '多少']
        factual_count = sum(1 for kw in factual_keywords if kw in query)
        if factual_count > 0:
            intent_scores['factual'] = factual_count * 1.5
            
        # 取最高分意图
        if not intent_scores:
            best_intent = 'default'
        else:
            best_intent = max(intent_scores, key=intent_scores.get)
            
        # 判断各意图是否激活（对应返回字段）
        is_exact_query = bool(intent_scores.get('precise', 0) > 0)
        is_troubleshooting = bool(intent_scores.get('fault', 0) > 0)
        is_comparison = bool(intent_scores.get('compare', 0) > 0)
        is_factual = bool(intent_scores.get('factual', 0) > 0)
            
        # 根据最高分意图返回对应的权重配置
        if best_intent == 'precise':
            # 代码/技术术语查询：偏好BM25精确匹配
            weights = {'vector': 0.5, 'bm25': 2.0, 'graph': 0.3}
        elif best_intent == 'fault':
            # 故障排查：偏好图谱关系推理
            weights = {'vector': 1.0, 'bm25': 0.8, 'graph': 1.5}
        elif best_intent == 'compare':
            # 对比查询：平衡向量语义和BM25精确
            weights = {'vector': 1.3, 'bm25': 1.2, 'graph': 0.8}
        elif best_intent == 'factual':
            # 事实查询：偏好向量语义相似度
            weights = {'vector': 1.5, 'bm25': 1.0, 'graph': 0.8}
        else:
            # 默认权重
            weights = {'vector': 1.2, 'bm25': 1.0, 'graph': 1.0}
            
        return {
            'is_exact_query': is_exact_query,
            'is_troubleshooting': is_troubleshooting,
            'is_comparison': is_comparison,
            'is_factual': is_factual,
            'weights': weights
        }
    
    def _semantic_deduplicate(self, results, threshold=0.92):
        """
        语义去重：基于向量相似度去除重复内容
        results: List[(doc, score)]
        threshold: 相似度阈值，超过则认为重复
        """
        if not results:
            return results
        
        unique_results = []
        seen_embeddings = []
        
        for doc, score in results:
            # 获取文档向量（使用缓存避免重复计算）
            doc_key = f"{doc.metadata.get('db_name')}_{doc.metadata.get('chunk_id')}"
            
            if doc_key in self._query_embedding_cache:
                doc_embedding = self._query_embedding_cache[doc_key]
            else:
                doc_embedding = self.embeddings.embed_query(doc.page_content[:1000])
                self._query_embedding_cache[doc_key] = doc_embedding
            
            # 检查是否与已有结果相似
            is_duplicate = False
            for seen_emb in seen_embeddings:
                similarity = np.dot(doc_embedding, seen_emb) / (
                    np.linalg.norm(doc_embedding) * np.linalg.norm(seen_emb) + 1e-8
                )
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append((doc, score))
                seen_embeddings.append(doc_embedding)
        
        return unique_results
    
    def _smart_truncate_text(self, text, max_tokens=400, preserve_ratio=0.3):
        """
        上下文动态截断：智能保留开头和关键信息
        preserve_ratio: 保留开头的比例
        """
        # 估算token数（中文字符约1:1到1:1.5）
        estimated_tokens = len(text) * 0.8
        
        if estimated_tokens <= max_tokens:
            return text
        
        # 计算保留长度
        max_chars = int(max_tokens / 0.8)
        head_len = int(max_chars * preserve_ratio)
        tail_len = max_chars - head_len - 10  # 预留省略号空间
        
        # 智能截断：保留开头和结尾
        head_text = text[:head_len]
        tail_text = text[-tail_len:] if len(text) > head_len + tail_len else ""
        
        if tail_text:
            return head_text + "\n... [内容省略] ...\n" + tail_text
        else:
            return head_text + "..."
    
    def search(self, query, top_k=5, db_names=['default']):
        """Hybrid Search: Vector + BM25 + Graph RAG across multiple databases."""
        if not isinstance(db_names, list):
            db_names = [db_names]
            
        all_vector_results = []
        all_bm25_results = []
        
        # 1. Graph RAG 检索（始终执行）
        graph_results = self.graph_search(query)
        
        # 2. 查询意图分析 - 动态调整策略
        intent = self._analyze_query_intent(query)
        weights = intent['weights']
        
        # 根据意图动态调整召回数量
        if intent['is_exact_query']:
            vector_k, bm25_k = 30, 80  # 精确查询：少向量，多BM25
        elif intent['is_troubleshooting']:
            vector_k, bm25_k = 50, 40  # 故障排查：平衡两者
        else:
            vector_k, bm25_k = 60, 50  # 默认：多向量，多BM25
        
        for db_name in db_names:
            vs = self.vector_stores.get(db_name)
            if not vs:
                # 尝试加载
                self.load_index(db_name)
                vs = self.vector_stores.get(db_name)
                
            if vs:
                try:
                    vector_results = vs.similarity_search_with_score(query, k=vector_k)
                    print(f"[Search Debug] Vector search for '{db_name}': {len(vector_results)} results")
                    all_vector_results.extend(vector_results)
                except Exception as e:
                    print(f"[Search Debug] Vector search error: {e}")
            else:
                print(f"[Search Debug] No vector store for '{db_name}'")
                
            # BM25 Search
            bm25_store = self.bm25_stores.get(db_name)
            if bm25_store and bm25_store.get("bm25"):
                bm25 = bm25_store["bm25"]
                chunk_map = bm25_store["chunk_map"]
                chunk_list = bm25_store.get("chunk_list", [])
                
                if not chunk_list:
                    print(f"[Search Debug] BM25 chunk_list empty for '{db_name}'")
                    continue
                
                print(f"[Search Debug] BM25 index for '{db_name}': {len(chunk_list)} chunks")
                
                # 使用优化的分词方法（jieba中文分词）
                tokenized_query = self._tokenize_for_bm25(query)
                print(f"[Search Debug] Tokenized query: {tokenized_query[:10]}...")
                
                doc_scores = bm25.get_scores(tokenized_query)
                non_zero_scores = [s for s in doc_scores if s > 0]
                print(f"[Search Debug] BM25 scores: {len(non_zero_scores)} non-zero, max={max(doc_scores) if len(doc_scores) > 0 else 0}")
                
                top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:bm25_k]
                
                # 使用chunk_list保持与doc_scores相同的顺序
                for idx in top_indices:
                    if doc_scores[idx] > 0 and idx < len(chunk_list):
                        chunk = chunk_list[idx]
                        
                        # 尝试获取 doc_name，为了防止 DetachedInstanceError，增加 app_context() 和 session 检查
                        doc_name = "Unknown"
                        try:
                            # 如果 chunk.document 已经被 eager load，直接获取
                            if hasattr(chunk, 'document') and chunk.document:
                                doc_name = chunk.document.doc_name
                        except Exception:
                            # 捕获 DetachedInstanceError 并使用备用查询
                            from app.db import db
                            from app.db.models import Document
                            from flask import current_app
                            try:
                                with current_app.app_context():
                                    doc = db.session.query(Document).filter_by(doc_id=chunk.doc_id).first()
                                    if doc:
                                        doc_name = doc.doc_name
                            except Exception:
                                pass
                                
                        meta = {
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "doc_name": doc_name,
                            "chunk_index": chunk.chunk_index,
                            "db_name": db_name
                        }
                        doc = LangchainDocument(page_content=chunk.content, metadata=meta)
                        all_bm25_results.append((doc, float(doc_scores[idx])))
                        
        # 3. 语义去重：对各自检索结果先去重（阈值设为0.95，只去掉真正重复的，保留相关但不同的）
        vector_before = len(all_vector_results)
        bm25_before = len(all_bm25_results)
        all_vector_results = self._semantic_deduplicate(all_vector_results, threshold=0.95)
        all_bm25_results = self._semantic_deduplicate(all_bm25_results, threshold=0.95)
        print(f"Semantic dedup: Vector {vector_before} -> {len(all_vector_results)}, BM25 {bm25_before} -> {len(all_bm25_results)}")
        
        # 4. 自适应 RRF Fusion (Reciprocal Rank Fusion) - 使用动态权重
        fused_scores = {}
        k_rrf = 60 # RRF constant
        
        # 使用意图分析得到的动态权重
        vector_weight = weights['vector']
        bm25_weight = weights['bm25']
        graph_weight = weights['graph']
        
        # 向量检索结果排序（分数降序，因为使用了normalize_embeddings=True，返回的是相似度分数）
        all_vector_results.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc, score) in enumerate(all_vector_results):
            doc_id = f"{doc.metadata.get('db_name')}_{doc.metadata.get('chunk_id')}"
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0, "sources": []}
            fused_scores[doc_id]["score"] += vector_weight * (1 / (k_rrf + rank + 1))
            fused_scores[doc_id]["sources"].append("vector")
            
        # BM25检索结果排序（分数降序）
        all_bm25_results.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc, score) in enumerate(all_bm25_results):
            doc_id = f"{doc.metadata.get('db_name')}_{doc.metadata.get('chunk_id')}"
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0, "sources": []}
            fused_scores[doc_id]["score"] += bm25_weight * (1 / (k_rrf + rank + 1))
            if "bm25" not in fused_scores[doc_id]["sources"]:
                fused_scores[doc_id]["sources"].append("bm25")
        
        # 图谱结果也加入融合（使用实际chunk_id避免重复）
        for rank, (doc, score) in enumerate(graph_results):
            # 使用文档的chunk_id或构建唯一ID
            doc_id = doc.metadata.get('chunk_id', f"graph_{rank}")
            # 统一ID前缀格式：如果已有graph_前缀则不再重复添加
            if not doc_id.startswith("graph_"):
                unique_doc_id = f"graph_{doc_id}"
            else:
                unique_doc_id = doc_id
            # 如果已存在，保留分数更高的版本
            if unique_doc_id in fused_scores:
                if score > fused_scores[unique_doc_id].get("original_score", 0):
                    fused_scores[unique_doc_id]["original_score"] = score
            else:
                fused_scores[unique_doc_id] = {"doc": doc, "score": 0.0, "sources": [], "original_score": score}
            fused_scores[unique_doc_id]["score"] += graph_weight * (1 / (k_rrf + rank + 1))
            if "graph" not in fused_scores[unique_doc_id]["sources"]:
                fused_scores[unique_doc_id]["sources"].append("graph")
            
        # Sort by fused score
        sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        print(f"RRF Fusion: {len(sorted_results)} unique results after fusion")
        
        # 多阶段检索：取融合后的 Top-N 进入重排序阶段
        # 优先保留来自多个源的结果（向量+BM25都命中的更可信）
        def result_priority(item):
            """结果优先级：加权模型，每多一个源+0.2分bonus"""
            source_count = len(set(item.get("sources", [])))
            source_bonus = (source_count - 1) * 0.2  # 每多一个源+0.2分bonus
            return item["score"] * (1 + source_bonus)
        
        sorted_results.sort(key=result_priority, reverse=True)
        
        # 取更多结果进入重排序，避免遗漏
        rerank_candidates = min(50, len(sorted_results))
        top_fused = [item["doc"] for item in sorted_results[:rerank_candidates]]
        print(f"Rerank stage: {len(top_fused)} candidates for reranking")
        
        # 提取图谱结果（已在融合中）
        graph_items = [item for item in sorted_results if "graph" in item.get("sources", [])]
        # 统一格式为 (doc, score) 元组，score 使用融合的 RR F 分数
        graph_docs = [(item["doc"], item["score"]) for item in graph_items[:10]]  # 最多保留10个图谱结果
        
        # Reranker 缓存机制 (简单的内存缓存，生产环境可用 Redis)
        if hasattr(self, 'reranker_cache') is False:
            self.reranker_cache = {}
            
        cache_key = f"{query}_" + "_".join(db_names)
        if cache_key in self.reranker_cache:
            print("Hit Reranker Cache!")
            cached_reranked = self.reranker_cache[cache_key]
            final_results = graph_docs + cached_reranked
            # 去重
            seen = set()
            dedup_results = []
            for d, s in final_results:
                content_hash = hash(d.page_content[:100])
                if content_hash not in seen:
                    seen.add(content_hash)
                    dedup_results.append((d, s))
            return dedup_results[:top_k]  # 返回 (doc, score) 元组列表
        
        # 5. Rerank (重排序)
        reranked_results = []
        if top_fused:
            # 上下文动态截断：智能保留关键信息，避免Token超限
            max_tokens = 600  # 约750字符
            truncated_docs = []
            for doc in top_fused:
                truncated_content = self._smart_truncate_text(
                    doc.page_content,
                    max_tokens=max_tokens,
                    preserve_ratio=0.4  # 保留40%开头
                )
                # 创建新文档对象（不修改原对象）
                truncated_doc = LangchainDocument(
                    page_content=truncated_content,
                    metadata=doc.metadata.copy()
                )
                truncated_docs.append(truncated_doc)
            
            pairs = [[query, doc.page_content] for doc in truncated_docs]
            with torch.no_grad():
                inputs = self.reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
            
            for doc, score in zip(top_fused, scores):
                chunk_index = doc.metadata.get("chunk_index")
                document_id = doc.metadata.get("doc_id")
                
                # Fetch surrounding chunks to provide broader context (Parent document simulation)
                # 跳过图谱节点（图谱节点没有chunk_index，为None）
                if chunk_index is not None and document_id is not None and not str(document_id).startswith("graph_"):
                    try:
                        from app.db.models import Chunk
                        from flask import current_app
                        from app.db import db
                        with current_app.app_context():
                            surrounding_chunks = db.session.query(Chunk).filter(
                                Chunk.doc_id == document_id,
                                Chunk.chunk_index >= max(0, chunk_index - 1),
                                Chunk.chunk_index <= chunk_index + 1
                            ).order_by(Chunk.chunk_index).all()
                            
                            if surrounding_chunks:
                                expanded_content = "\n...\n".join([c.content for c in surrounding_chunks])
                                # 创建新doc避免修改原doc，因为原doc可能同时在graph_docs中
                                doc = LangchainDocument(
                                    page_content=expanded_content,
                                    metadata=doc.metadata.copy()
                                )
                    except Exception as e:
                        pass

                reranked_results.append((doc, score.item()))
                
            # 按重排序得分降序
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            print(f"Reranking complete: {len(reranked_results)} results, top score: {reranked_results[0][1] if reranked_results else 0}")
            
            # 缓存重排结果
            self.reranker_cache[cache_key] = reranked_results[:top_k * 2]
            
            # 智能合并结果：去重并保留多源证据
            seen_content = set()
            final_results = []
            
            # 1. 优先添加图谱结果（如果有）- graph_docs 已经是 (doc, score) 格式
            for doc, score in graph_docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    final_results.append((doc, score))
            
            # 2. 添加重排序结果（去重）
            for doc, score in reranked_results:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    final_results.append((doc, score))
                if len(final_results) >= top_k * 3:  # 保留更多候选
                    break
            
            # 3. 如果结果太少，从融合结果中补充（统一为 (doc, score) 格式）
            if len(final_results) < top_k:
                for item in sorted_results:
                    doc = item["doc"]
                    score = item["score"]
                    content_hash = hash(doc.page_content[:100])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        final_results.append((doc, score))
                    if len(final_results) >= top_k:
                        break
            
            # 截断到最终需要的数量
            final_results = final_results[:top_k]
            
        else:
            # 没有重排序候选，使用融合结果（统一为 (doc, score) 格式）
            print(f"No reranking candidates, using fused results")
            final_results = [(item["doc"], item["score"]) for item in sorted_results[:top_k]]
        
        print(f"Search complete: returning {len(final_results)} results")
        
        # 如果所有检索都为空，返回空列表
        if not final_results:
            print("[Search Warning] All retrieval methods returned empty results")
        
        return final_results
