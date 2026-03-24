import os
import sys

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
        
        self.vector_store = None
        self.bm25 = None
        self.chunk_map = {} # Map chunk_id to Chunk object for BM25
        
        # 为了支持多库，我们需要维护多个 FAISS 索引实例和 BM25 实例
        # 结构: {"db_name": vector_store}
        self.vector_stores = {}
        # 结构: {"db_name": {"bm25": bm25_instance, "chunk_map": {}}}
        self.bm25_stores = {}
        
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
            print(f"No existing FAISS index found for '{db_name}'.")
            self.vector_stores[db_name] = None
            
        self.build_bm25_index(db_name)

    def build_bm25_index(self, db_name='default'):
        """Build BM25 index from DB chunks for specific db_name."""
        # 需要联表查询属于该库的 chunks
        # 这里有可能会在无上下文时调用，我们通过 db.session 检查或者简单地直接查询
        try:
            chunks = Chunk.query.join(Document).filter(Document.db_name == db_name).all()
            if not chunks:
                self.bm25_stores[db_name] = {"bm25": None, "chunk_map": {}}
                return
                
            tokenized_corpus = []
            chunk_map = {}
            for chunk in chunks:
                # 使用重排序模型的 tokenizer 替代 jieba 进行分词
                tokens = self.reranker_tokenizer.tokenize(chunk.content)
                tokenized_corpus.append(tokens)
                chunk_map[chunk.chunk_id] = chunk
                
            bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_stores[db_name] = {"bm25": bm25, "chunk_map": chunk_map}
            print(f"Built BM25 index for '{db_name}' with {len(chunks)} chunks.")
        except Exception as e:
            print(f"Error building BM25 index for {db_name}: {e}")
            self.bm25_stores[db_name] = {"bm25": None, "chunk_map": {}}

    def build_index(self, db_name='default'):
        """Build index from all chunks in specific DB."""
        chunks = Chunk.query.join(Document).filter(Document.db_name == db_name).all()
        if not chunks:
            print(f"No chunks to index for '{db_name}'.")
            # 如果该库被清空了，清理内存中的实例并删除磁盘上的旧索引文件
            if db_name in self.vector_stores:
                del self.vector_stores[db_name]
            self.bm25_stores[db_name] = {"bm25": None, "chunk_map": {}}
            db_index_path = self.get_index_path(db_name)
            faiss_file = os.path.join(db_index_path, "index.faiss")
            pkl_file = os.path.join(db_index_path, "index.pkl")
            if os.path.exists(faiss_file):
                os.remove(faiss_file)
            if os.path.exists(pkl_file):
                os.remove(pkl_file)
            return

        documents = []
        for chunk in chunks:
            meta = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "db_name": db_name
            }
            doc = LangchainDocument(page_content=chunk.content, metadata=meta)
            documents.append(doc)

        print(f"Creating index for '{db_name}' with {len(documents)} documents...")
        vs = FAISS.from_documents(documents, self.embeddings)
        self.vector_stores[db_name] = vs
        
        db_index_path = self.get_index_path(db_name)
        vs.save_local(db_index_path)
        print(f"Index saved to {db_index_path}")
        
        # 兼容旧代码，将默认库的指标记录下来
        if db_name == 'default':
            new_index = VectorIndex(chunk_count=len(documents))
            db.session.add(new_index)
            db.session.commit()
        
        self.build_bm25_index(db_name)

    def add_documents(self, chunks, db_name='default'):
        """Incremental update (add new chunks to specific db)."""
        vs = self.vector_stores.get(db_name)
        if not vs:
            self.build_index(db_name)
            return

        documents = []
        for chunk in chunks:
            meta = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
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

    def search(self, query, top_k=5, db_names=['default']):
        """Hybrid Search: Vector + BM25 across multiple databases."""
        if not isinstance(db_names, list):
            db_names = [db_names]
            
        all_vector_results = []
        all_bm25_results = []
        
        for db_name in db_names:
            vs = self.vector_stores.get(db_name)
            if not vs:
                # 尝试加载
                self.load_index(db_name)
                vs = self.vector_stores.get(db_name)
                
            if vs:
                vector_results = vs.similarity_search_with_score(query, k=top_k)
                all_vector_results.extend(vector_results)
                
            # BM25 Search
            bm25_store = self.bm25_stores.get(db_name)
            if bm25_store and bm25_store.get("bm25"):
                bm25 = bm25_store["bm25"]
                chunk_map = bm25_store["chunk_map"]
                
                tokenized_query = self.reranker_tokenizer.tokenize(query)
                doc_scores = bm25.get_scores(tokenized_query)
                top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
                
                chunk_ids = list(chunk_map.keys())
                for idx in top_indices:
                    if doc_scores[idx] > 0:
                        chunk_id = chunk_ids[idx]
                        chunk = chunk_map[chunk_id]
                        meta = {
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "chunk_index": chunk.chunk_index,
                            "db_name": db_name
                        }
                        doc = LangchainDocument(page_content=chunk.content, metadata=meta)
                        all_bm25_results.append((doc, float(doc_scores[idx])))
                        
        # 3. RRF Fusion (Reciprocal Rank Fusion)
        fused_scores = {}
        k = 60 # RRF constant
        
        # 为了进行跨库 RRF 融合，我们需要对所有向量结果和所有 BM25 结果进行一次全局排序
        # 向量距离越小越好
        all_vector_results.sort(key=lambda x: x[1])
        for rank, (doc, score) in enumerate(all_vector_results):
            doc_id = f"{doc.metadata.get('db_name')}_{doc.metadata.get('chunk_id')}"
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0}
            fused_scores[doc_id]["score"] += 1 / (k + rank + 1)
            
        # BM25 分数越大越好
        all_bm25_results.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc, score) in enumerate(all_bm25_results):
            doc_id = f"{doc.metadata.get('db_name')}_{doc.metadata.get('chunk_id')}"
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0}
            fused_scores[doc_id]["score"] += 1 / (k + rank + 1)
            
        # Sort by fused score
        sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # 取融合后的Top-50进入重排序阶段
        top_50_fused = [item["doc"] for item in sorted_results[:50]]
        
        # 4. Rerank (重排序)
        reranked_results = []
        if top_50_fused:
            pairs = [[query, doc.page_content] for doc in top_50_fused]
            with torch.no_grad():
                inputs = self.reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
            
            for doc, score in zip(top_50_fused, scores):
                reranked_results.append((doc, score.item()))
                
            # 按重排序得分降序
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            final_results = reranked_results[:top_k]
        else:
            final_results = []
        
        return final_results
