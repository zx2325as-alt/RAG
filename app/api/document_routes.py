"""
文档管理相关路由：上传、删除、查询
"""
from flask import request, jsonify, current_app
from app.db.models import Document, Chunk
from app.db import db
from app.api.common import api_bp, get_document_service, get_qa_service, get_kb_service
import os
import threading
import shutil


def process_document_background(app, doc_id, filepath, db_name):
    """后台任务处理文档，避免阻塞 API"""
    with app.app_context():
        try:
            document_service = get_document_service()
            kb = get_kb_service()

            chunks = document_service.parse_document(doc_id, filepath)

            if chunks:
                kb.add_documents(chunks, db_name=db_name)
        except Exception as e:
            import traceback
            app.logger.error(f"Background processing failed for doc {doc_id}: {e}\n{traceback.format_exc()}")
            doc = Document.query.get(doc_id)
            if doc:
                doc.status = 'failed'
                db.session.commit()


@api_bp.route('/upload', methods=['POST'])
def upload_document():
    if 'files[]' not in request.files:
        current_app.logger.error('Upload failed: No file part in request')
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('files[]')
    db_name = request.form.get('db_name', 'default')

    if not files or files[0].filename == '':
        current_app.logger.error('Upload failed: No selected file')
        return jsonify({'error': 'No selected file'}), 400

    uploaded_files = []
    errors = []

    document_service = get_document_service()

    for file in files:
        current_app.logger.info(f'Start uploading document: {file.filename} to db: {db_name}')
        try:
            doc = document_service.upload_document(file, db_name=db_name)
            current_app.logger.info(f'Document {doc.doc_name} uploaded successfully with id {doc.doc_id}')

            filepath = os.path.join(document_service.upload_folder, db_name, doc.doc_name)
            thread = threading.Thread(
                target=process_document_background,
                args=(current_app._get_current_object(), doc.doc_id, filepath, db_name)
            )
            thread.start()

            uploaded_files.append(doc.doc_name)
        except Exception as e:
            current_app.logger.error(f'Upload failed for {file.filename}: {str(e)}')
            errors.append(f"{file.filename}: {str(e)}")

    if errors:
        return jsonify({'error': 'Partial success', 'uploaded': uploaded_files, 'failed': errors}), 207

    return jsonify({
        'message': 'All files uploaded successfully',
        'filenames': uploaded_files
    })


@api_bp.route('/documents', methods=['GET'])
def get_documents():
    docs = Document.query.all()
    result = []
    for doc in docs:
        result.append({
            'doc_id': doc.doc_id,
            'doc_name': doc.doc_name,
            'status': doc.status,
            'db_name': getattr(doc, 'db_name', 'default'),
            'upload_time': doc.upload_time.strftime("%Y-%m-%d %H:%M:%S") if doc.upload_time else None
        })
    return jsonify({'documents': result})


@api_bp.route('/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    try:
        doc = Document.query.get(doc_id)
        if not doc:
            return jsonify({'error': 'Document not found in database'}), 404

        db_name = getattr(doc, 'db_name', 'default')

        try:
            document_service = get_document_service()
            filepath = os.path.join(document_service.upload_folder, db_name, doc.doc_name)
            if os.path.exists(filepath):
                os.remove(filepath)
            else:
                current_app.logger.warning(f"File {filepath} not found on disk, proceeding with DB deletion.")
        except Exception as file_e:
            current_app.logger.warning(f"Failed to delete physical file: {file_e}")

        Chunk.query.filter_by(doc_id=doc_id).delete()
        db.session.delete(doc)
        db.session.commit()

        try:
            kb = get_kb_service()
            kb.build_index(db_name)
        except Exception as kb_e:
            current_app.logger.warning(f"Failed to rebuild index after deletion: {kb_e}")

        return jsonify({'message': 'Document deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/documents/clear_all', methods=['DELETE'])
def clear_all_documents():
    """清空所有文档、文本块记录及本地物理文件和向量索引"""
    try:
        from app.db.models import QueryLog, QACache, VectorIndex
        from app.config import Config

        db.session.query(Chunk).delete()
        db.session.query(Document).delete()
        db.session.query(QueryLog).delete()
        db.session.query(QACache).delete()
        db.session.query(VectorIndex).delete()
        db.session.commit()

        document_service = get_document_service()
        if os.path.exists(document_service.upload_folder):
            shutil.rmtree(document_service.upload_folder)
            os.makedirs(document_service.upload_folder, exist_ok=True)

        index_dir = Config.FAISS_INDEX_PATH
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
            os.makedirs(index_dir, exist_ok=True)

        kb = get_kb_service()
        kb.vector_stores = {}
        kb.bm25 = None
        kb.chunk_map = {}
        kb.bm25_stores = {}
        kb.build_index('default')

        qa = get_qa_service()
        if hasattr(qa, 'use_redis') and qa.use_redis:
            try:
                qa.redis_client.flushdb()
            except Exception:
                pass

        current_app.logger.info('All documents, databases, local files and indexes have been thoroughly cleared.')
        return jsonify({'message': 'All data cleared successfully'})
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f'Error clearing all data: {str(e)}')
        return jsonify({'error': str(e)}), 500
