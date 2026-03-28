import os
import hashlib
import logging
from werkzeug.utils import secure_filename
from app.config import Config
from app.db import db
from app.db.models import Document, Chunk
from app.utils.file_processor import process_file
from app.utils.text_processor import clean_text, chunk_text

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.upload_folder = Config.UPLOAD_FOLDER
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)

    def calculate_md5(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def upload_document(self, file, db_name='default'):
        # 修正：werkzeug 的 secure_filename 会过滤掉中文，导致 "97_人工智能.pdf" 变成 "97_.pdf"
        # 针对中文文件，我们可以将非 ASCII 字符保留，或者直接用原名但替换掉危险字符
        original_filename = file.filename
        # 简单安全处理：替换掉路径遍历符号，保留中文
        safe_filename = original_filename.replace('/', '_').replace('\\', '_').replace('..', '_')
        
        # 为不同数据库创建子目录
        db_folder = os.path.join(self.upload_folder, db_name)
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)
            
        filepath = os.path.join(db_folder, safe_filename)
        file.save(filepath)
        
        # Deduplication Check
        file_md5 = self.calculate_md5(filepath)
        existing_doc = Document.query.filter_by(doc_name=safe_filename, db_name=db_name).first() 
        # Note: In real production, we should add an md5 column to Document table and query by that.
        # For now, we check by filename as a simple proxy, or we can assume content deduplication is enough.
        # Let's stick to filename for simplicity unless we migrate schema.
        
        if existing_doc:
            # Check if status is completed, if so, maybe return existing
            if existing_doc.status == 'completed':
                logger.info(f"Document {safe_filename} already exists in {db_name}. Skipping processing.")
                os.remove(filepath) # Remove duplicate file
                return existing_doc
        
        # Create DB record
        new_doc = Document(doc_name=safe_filename, doc_type=safe_filename.split('.')[-1], status='pending', db_name=db_name)
        db.session.add(new_doc)
        db.session.commit()
        
        # Trigger parsing immediately (for demo purposes)
        # 已经在 API 层改为异步触发，这里只需返回创建好的文档记录即可
        return new_doc

    def parse_document(self, doc_id, filepath, progress_callback=None):
        doc = Document.query.get(doc_id)
        if not doc:
            return []

        doc.status = 'processing'
        db.session.commit()

        try:
            import logging
            from flask import current_app
            logger = current_app.logger if current_app else logging.getLogger(__name__)
            
            logger.info(f"[{doc.doc_name}] Starting to parse document (ID: {doc_id})...")
            
            # 1. Extract Text
            if progress_callback: progress_callback(25, "正在提取原始文本...")
            raw_text = process_file(filepath)
            logger.info(f"[{doc.doc_name}] Text extracted. Total Length: {len(raw_text)} characters.")
            
            # 2. Clean Text
            if progress_callback: progress_callback(35, "正在清洗文本...")
            cleaned_text = clean_text(raw_text)
            
            # 3. Chunk Text
            if progress_callback: progress_callback(45, "正在进行语义分块...")
            is_markdown = filepath.lower().endswith('.md')
            text_chunks = chunk_text(cleaned_text, is_markdown=is_markdown)
            logger.info(f"[{doc.doc_name}] Text chunked successfully into {len(text_chunks)} segments.")
            
            # 4. Save Chunks using bulk_save_objects for performance
            if progress_callback: progress_callback(55, f"正在将 {len(text_chunks)} 个分片存入数据库...")
            db_chunks = []
            for i, content in enumerate(text_chunks):
                chunk = Chunk(
                    doc_id=doc.doc_id,
                    content=content,
                    chunk_index=i
                )
                db_chunks.append(chunk)
                
            if db_chunks:
                db.session.bulk_save_objects(db_chunks)
            
            # 5. Extract and build Graph RAG (Intelligent Graph)
            try:
                from app.services.graph_service import GraphService
                graph_service = GraphService()
                logger.info(f"[{doc.doc_name}] Starting LLM-based Graph extraction...")
                if progress_callback: progress_callback(75, "正在利用 LLM 提取实体关系图谱...")
                # We use the generated db_chunks (which have doc_id but maybe not chunk_id yet, but content is there)
                graph_service.extract_and_store_graph(db_chunks, doc.doc_id)
                logger.info(f"[{doc.doc_name}] Graph extraction completed.")
            except Exception as e:
                logger.error(f"[{doc.doc_name}] Graph extraction failed: {e}")
            
            doc.status = 'completed'
            db.session.commit()
            
            # Since bulk_save_objects doesn't populate chunk_id automatically, we need to query them back
            saved_chunks = Chunk.query.filter_by(doc_id=doc.doc_id).order_by(Chunk.chunk_index).all()
            
            logger.info(f"[{doc.doc_name}] Processing and database insertion completed. {len(saved_chunks)} chunks saved.")
            return saved_chunks
            
        except Exception as e:
            doc.status = 'failed'
            db.session.commit()
            raise e
