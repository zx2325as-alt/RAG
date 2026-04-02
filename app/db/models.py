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


# ==================== 模型训练评估系统 ====================

class TrainingJob(db.Model):
    """训练任务记录"""
    __tablename__ = 'training_job'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_name = db.Column(db.String(255), nullable=False, index=True)
    status = db.Column(db.String(50), default='pending')  # pending/running/completed/failed
    config = db.Column(db.JSON, default=dict)
    
    # 时间记录
    start_time = db.Column(db.DateTime, nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)
    
    # 训练指标摘要
    metrics_summary = db.Column(db.JSON, default=dict)
    # {
    #   "final_loss": 0.5,
    #   "total_steps": 5000,
    #   "epochs": 3.0,
    #   "train_runtime": 3600
    # }
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelEvaluation(db.Model):
    """模型评估记录"""
    __tablename__ = 'model_evaluation'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_name = db.Column(db.String(255), nullable=False, index=True)
    eval_dataset = db.Column(db.String(255), nullable=False)
    sample_count = db.Column(db.Integer, default=0)
    
    # 评估指标
    metrics = db.Column(db.JSON, default=dict)
    # {
    #   "rouge_l": 0.85,
    #   "bleu": 0.72,
    #   "bertscore": 0.91,
    #   "perplexity": 12.5
    # }
    
    # 效率指标
    efficiency_metrics = db.Column(db.JSON, default=dict)
    # {
    #   "inference_speed": 45.2,  # tokens/sec
    #   "peak_memory_mb": 8192,
    #   "avg_memory_mb": 6144
    # }
    
    # 样例对比
    sample_comparisons = db.Column(db.JSON, default=list)
    # [{
    #   "input": "...",
    #   "reference": "...",
    #   "prediction": "...",
    #   "rouge_score": 0.9
    # }]
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ModelComparison(db.Model):
    """多模型对比记录"""
    __tablename__ = 'model_comparison'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    comparison_name = db.Column(db.String(255), nullable=False)
    model_names = db.Column(db.JSON, default=list)  # ["model1", "model2", ...]
    eval_dataset = db.Column(db.String(255), nullable=False)
    
    # 各模型指标
    model_metrics = db.Column(db.JSON, default=dict)
    # {
    #   "model1": {"rouge_l": 0.85, "bleu": 0.72, "train_time": 3600},
    #   "model2": {"rouge_l": 0.88, "bleu": 0.75, "train_time": 4200}
    # }
    
    # 样例对比
    sample_comparisons = db.Column(db.JSON, default=list)
    # [{
    #   "input": "问题",
    #   "reference": "参考答案",
    #   "model_outputs": {
    #       "model1": {"output": "...", "score": 0.9},
    #       "model2": {"output": "...", "score": 0.85}
    #   }
    # }]
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
