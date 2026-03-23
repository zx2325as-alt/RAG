import os
import hashlib
from werkzeug.utils import secure_filename
from app.config import Config
from app.db import db
from app.db.models import Document, Chunk
from app.utils.file_processor import process_file
from app.utils.text_processor import clean_text, chunk_text

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
        filename = secure_filename(file.filename)
        # 为不同数据库创建子目录
        db_folder = os.path.join(self.upload_folder, db_name)
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)
            
        filepath = os.path.join(db_folder, filename)
        file.save(filepath)
        
        # Deduplication Check
        file_md5 = self.calculate_md5(filepath)
        existing_doc = Document.query.filter_by(doc_name=filename, db_name=db_name).first() 
        # Note: In real production, we should add an md5 column to Document table and query by that.
        # For now, we check by filename as a simple proxy, or we can assume content deduplication is enough.
        # Let's stick to filename for simplicity unless we migrate schema.
        
        if existing_doc:
            # Check if status is completed, if so, maybe return existing
            if existing_doc.status == 'completed':
                print(f"Document {filename} already exists in {db_name}. Skipping processing.")
                os.remove(filepath) # Remove duplicate file
                return existing_doc
        
        # Create DB record
        new_doc = Document(doc_name=filename, doc_type=filename.split('.')[-1], status='pending', db_name=db_name)
        db.session.add(new_doc)
        db.session.commit()
        
        # Trigger parsing immediately (for demo purposes)
        try:
            # 获取解析的 chunks 以便进行增量更新
            chunks = self.parse_document(new_doc.doc_id, filepath)
            
            # 解析完成后，进行增量向量索引更新，避免每次都全量重建（极大提升速度）
            if chunks:
                from app.services.knowledge_base_service import KnowledgeBaseService
                kb = KnowledgeBaseService()
                kb.add_documents(chunks, db_name=db_name)
                
        except Exception as e:
            import traceback
            print(f"Parsing failed: {e}\n{traceback.format_exc()}")
            new_doc.status = 'failed'
            db.session.commit()

        return new_doc

    def parse_document(self, doc_id, filepath):
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
            raw_text = process_file(filepath)
            logger.info(f"[{doc.doc_name}] Text extracted. Total Length: {len(raw_text)} characters.")
            
            # 2. Clean Text
            cleaned_text = clean_text(raw_text)
            
            # 3. Chunk Text
            is_markdown = filepath.lower().endswith('.md')
            text_chunks = chunk_text(cleaned_text, is_markdown=is_markdown)
            logger.info(f"[{doc.doc_name}] Text chunked successfully into {len(text_chunks)} segments.")
            
            # 4. Save Chunks
            db_chunks = []
            for i, content in enumerate(text_chunks):
                chunk = Chunk(
                    doc_id=doc.doc_id,
                    content=content,
                    chunk_index=i
                )
                db.session.add(chunk)
                db_chunks.append(chunk)
            
            doc.status = 'completed'
            db.session.commit()
            logger.info(f"[{doc.doc_name}] Processing and database insertion completed.")
            return db_chunks
            
        except Exception as e:
            doc.status = 'failed'
            db.session.commit()
            raise e
