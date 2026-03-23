from datetime import datetime
from app.db import db

class Document(db.Model):
    __tablename__ = 'document'
    doc_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    doc_name = db.Column(db.String(255), nullable=False)
    doc_type = db.Column(db.String(50), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='pending')  # pending, processing, completed, failed
    db_name = db.Column(db.String(100), default='default') # 用于区分不同的知识库
    
    chunks = db.relationship('Chunk', backref='document', lazy=True)

class Chunk(db.Model):
    __tablename__ = 'chunk'
    chunk_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    doc_id = db.Column(db.Integer, db.ForeignKey('document.doc_id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)

class VectorIndex(db.Model):
    __tablename__ = 'vector_index'
    index_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    build_time = db.Column(db.DateTime, default=datetime.utcnow)
    chunk_count = db.Column(db.Integer, default=0)

class QueryLog(db.Model):
    __tablename__ = 'query_log'
    log_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(50), nullable=True)
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=True)
    latency = db.Column(db.Float, default=0.0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class QACache(db.Model):
    __tablename__ = 'qa_cache'
    cache_key = db.Column(db.String(32), primary_key=True)  # MD5 hash
    response = db.Column(db.Text, nullable=False)
    expire_time = db.Column(db.DateTime, nullable=False)
