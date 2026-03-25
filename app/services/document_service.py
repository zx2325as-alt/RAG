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
        # 已经在 API 层改为异步触发，这里只需返回创建好的文档记录即可
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
            
            # 4. Save Chunks using bulk_save_objects for performance
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
