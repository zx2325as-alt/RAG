from celery import Celery
import os
import time

# 配置 Celery，使用 RabbitMQ 作为 broker，Redis 作为 backend
broker_url = os.getenv('CELERY_BROKER_URL', 'amqp://guest:guest@localhost:5672//')
result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

celery_app = Celery(
    'doc_parser',
    broker=broker_url,
    backend=result_backend
)

@celery_app.task(name='parse_document_task')
def parse_document_task(doc_id: str, file_path: str):
    """
    异步解析文档任务：
    1. 提取文本 (PyMuPDF / PaddleOCR)
    2. 文本清洗
    3. 分块 (RecursiveCharacterTextSplitter)
    4. 写入数据库
    5. 发送向量化任务消息
    """
    print(f"开始解析文档任务: doc_id={doc_id}, file_path={file_path}")
    
    try:
        # 模拟耗时的解析过程
        time.sleep(2)
        print(f"文档 {doc_id} 文本提取完成")
        
        # 模拟分块
        chunks = ["chunk1", "chunk2", "chunk3"]
        print(f"文档 {doc_id} 分块完成，共 {len(chunks)} 块")
        
        # 模拟写入数据库
        # db.save(chunks)
        
        # 触发向量化任务 (假设有另一个 worker 监听此任务)
        celery_app.send_task('vectorize_chunks_task', args=[doc_id, chunks])
        
        print(f"文档 {doc_id} 解析任务成功完成！")
        return {"status": "success", "doc_id": doc_id, "chunk_count": len(chunks)}
        
    except Exception as e:
        print(f"解析文档 {doc_id} 失败: {e}")
        return {"status": "failed", "error": str(e)}