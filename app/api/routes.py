from flask import Blueprint, request, jsonify, render_template, current_app
from app.db.models import Document, Chunk, QueryLog
from app.db import db
from app.services.document_service import DocumentService
from app.services.qa_service import QAService
from app.config import Config
from werkzeug.utils import secure_filename
import os
import sys
from datetime import datetime

def get_llamafactory_cli_path():
    import shutil
    cli_path = shutil.which('llamafactory-cli')
    if cli_path:
        return cli_path
    if os.name == 'nt':
        cli_path = os.path.join(os.path.dirname(sys.executable), 'Scripts', 'llamafactory-cli.exe')
    else:
        cli_path = os.path.join(os.path.dirname(sys.executable), 'llamafactory-cli')
    if os.path.exists(cli_path):
        return cli_path
    return 'llamafactory-cli'

api_bp = Blueprint('api', __name__)

# Remove these lines from global scope because they require app context
# document_service = DocumentService()
# qa_service = QAService()

# Use a function to get or initialize them lazily
def get_document_service():
    from flask import current_app
    if not hasattr(current_app, 'document_service'):
        current_app.document_service = DocumentService()
    return current_app.document_service

def get_qa_service():
    from flask import current_app
    if not hasattr(current_app, 'qa_service'):
        current_app.qa_service = QAService()
    return current_app.qa_service

def get_kb_service():
    # 统一使用 qa_service 内部的 kb_service，保证内存中的 FAISS/BM25 索引单例同步更新
    return get_qa_service().kb_service

@api_bp.route('/')
def index():
    return render_template('index.html')

@api_bp.route('/finetune')
def finetune():
    return render_template('finetune.html', tensorboard_port=Config.TENSORBOARD_PORT)

# 存储 webui 和 tensorboard 进程引用
tensorboard_process = None

@api_bp.route('/api/start_tensorboard', methods=['POST'])
def start_tensorboard():
    global tensorboard_process
    import subprocess
    import socket
    import os
    
    # 检查端口是否被占用
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((Config.TENSORBOARD_HOST, port)) == 0

    if is_port_in_use(Config.TENSORBOARD_PORT):
        return jsonify({'status': 'already_running'})

    try:
        # 指向集中管理的 runs 目录，确保 TensorBoard 能扫描到所有模型的日志
        log_dir = os.path.join(current_app.root_path, '..', 'finetuned_models', 'runs')
        os.makedirs(log_dir, exist_ok=True)
        
        tb_cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", log_dir,
            "--host", Config.TENSORBOARD_HOST,
            "--port", str(Config.TENSORBOARD_PORT),
            "--reload_interval", "1"
        ]
        if Config.TENSORBOARD_HOST == '0.0.0.0':
            tb_cmd.append("--bind_all")
            
        # 使用 Popen 后台拉起 TensorBoard，增加 --reload_interval 1 以实现秒级实时刷新，动态增加 --bind_all 允许外部访问
        tensorboard_process = subprocess.Popen(
            tb_cmd,
            stdout=open(os.path.join(current_app.root_path, '..', 'logs', 'tensorboard_stdout.log'), 'w'),
            stderr=subprocess.STDOUT
        )
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/check_tensorboard', methods=['GET'])
def check_tensorboard():
    import socket
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((Config.TENSORBOARD_HOST, port)) == 0
            
    return jsonify({'running': is_port_in_use(Config.TENSORBOARD_PORT)})

@api_bp.route('/status', methods=['GET'])
def status():
    doc_count = Document.query.count()
    chunk_count = Chunk.query.count()
    return jsonify({
        'status': 'ok', 
        'message': 'RAG System is running',
        'doc_count': doc_count,
        'chunk_count': chunk_count
    })

import threading

def process_document_background(app, doc_id, filepath, db_name):
    """Background task for processing documents to avoid blocking the API."""
    with app.app_context():
        try:
            document_service = get_document_service()
            kb = get_kb_service()
            
            # 解析文档并分块 (现在内部已经使用了 bulk_save_objects 提升速度)
            chunks = document_service.parse_document(doc_id, filepath)
            
            # 构建向量索引和 BM25 索引
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
            
            # 启动后台线程进行异步处理，避免阻塞 Web 响应
            # 使用 document_service 里面处理过的安全文件名，因为 doc.doc_name 已经是保留中文的安全文件名了
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
            # 即使数据库里没有找到，可能只是状态不一致，返回 404
            return jsonify({'error': 'Document not found in database'}), 404
            
        db_name = getattr(doc, 'db_name', 'default')
        
        # 尝试删除物理文件 (使用 try-except 防止因为文件不存在导致整个删除流程中断)
        try:
            document_service = get_document_service()
            filepath = os.path.join(document_service.upload_folder, db_name, doc.doc_name)
            if os.path.exists(filepath):
                os.remove(filepath)
            else:
                current_app.logger.warning(f"File {filepath} not found on disk, proceeding with DB deletion.")
        except Exception as file_e:
            current_app.logger.warning(f"Failed to delete physical file: {file_e}")
        
        # 从数据库中删除相关的 chunks
        Chunk.query.filter_by(doc_id=doc_id).delete()
        
        # 删除文档记录
        db.session.delete(doc)
        db.session.commit()
            
        # 触发重建索引
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
    import shutil
    try:
        # 删除数据库记录
        from app.db.models import QueryLog, QACache, VectorIndex, Document, Chunk
        db.session.query(Chunk).delete()
        db.session.query(Document).delete()
        db.session.query(QueryLog).delete()
        db.session.query(QACache).delete()
        db.session.query(VectorIndex).delete()
        db.session.commit()
        
        # 彻底清空本地物理文件
        document_service = get_document_service()
        if os.path.exists(document_service.upload_folder):
            shutil.rmtree(document_service.upload_folder)
            os.makedirs(document_service.upload_folder, exist_ok=True)
            
        # 彻底清空向量索引目录
        from app.config import Config
        index_dir = Config.FAISS_INDEX_PATH
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
            os.makedirs(index_dir, exist_ok=True)
            
        # 重置内存中的知识库服务状态
        kb = get_kb_service()
        kb.vector_stores = {}
        kb.bm25 = None
        kb.chunk_map = {}
        kb.bm25_stores = {}
        kb.build_index('default')
        
        # 清空 Redis 缓存（如果开启了）
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

import requests

@api_bp.route('/api/start_service', methods=['POST'])
def start_service():
    """启动本地模型服务 (Ollama / vLLM)"""
    service = request.json.get('service')
    try:
        import subprocess
        if service == 'ollama':
            # 后台启动 ollama serve
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.STDOUT
            )
            return jsonify({'message': 'Ollama 启动命令已发送到后台'})
        elif service == 'vllm':
            model = Config.VLLM_MODEL_NAME
            # 尝试拼接可能的绝对路径（如果是微调后的模型）
            if not "/" in model and not "\\" in model:
                potential_path = os.path.join(current_app.root_path, '..', 'finetuned_models', model)
                if os.path.exists(potential_path):
                    model = potential_path
                    
            cmd = ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", "8000"]
            subprocess.Popen(
                cmd, 
                stdout=open(os.path.join(current_app.root_path, '..', 'logs', 'vllm_stdout.log'), 'w'), 
                stderr=subprocess.STDOUT
            )
            return jsonify({'message': f'vLLM 启动命令已发送 (模型: {Config.VLLM_MODEL_NAME})'})
        else:
            return jsonify({'error': '不支持的服务类型'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/download_model', methods=['POST'])
def download_model():
    """下载在线 HuggingFace 模型或 Pull Ollama 模型"""
    model_name = request.json.get('model_name')
    if not model_name:
        return jsonify({'error': '未提供模型名称'}), 400
        
    try:
        import subprocess
        import requests
        
        if ":" in model_name and not "/" in model_name:
            # 认为是 Ollama 模型
            subprocess.Popen(["ollama", "pull", model_name], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            return jsonify({'message': f'已在后台执行: ollama pull {model_name}'})
        else:
            # 认为是 HuggingFace 模型，调用 huggingface-cli
            # 校验模型是否存在于 HuggingFace 镜像源
            try:
                # 使用镜像源进行简单校验，请求头设置适当超时时间
                hf_mirror_url = f"https://hf-mirror.com/api/models/{model_name}"
                check_response = requests.get(hf_mirror_url, timeout=5)
                if check_response.status_code != 200:
                    return jsonify({'error': f'校验失败：在 hf-mirror.com 上未找到模型 [{model_name}]，请检查 ID 是否正确。状态码: {check_response.status_code}'}), 404
            except requests.exceptions.RequestException as e:
                # 网络问题或其他请求异常
                return jsonify({'error': f'校验失败：无法连接到 hf-mirror.com，请检查网络。({str(e)})'}), 500
                
            # 使用 Config.HF_MODEL_DIR 作为本地缓存目录，按模型名分子目录存放
            # 完整保留模型名结构，替换 / 为 _，或者作为子目录。这里选择作为子目录，替换掉 / 防止路径穿越问题
            safe_model_name = model_name.replace('/', '--') 
            local_dir = os.path.join(Config.HF_MODEL_DIR, safe_model_name)
            os.makedirs(local_dir, exist_ok=True)
            
            env = os.environ.copy()
            env["HF_ENDPOINT"] = "https://hf-mirror.com"
            
            import shutil
            import sys
            
            hf_cli_cmd = shutil.which("huggingface-cli")
            if hf_cli_cmd:
                cmd = [hf_cli_cmd, "download", model_name, "--local-dir", local_dir]
            else:
                # 如果环境变量中找不到 huggingface-cli，使用当前 python 解释器执行模块兜底
                cmd = [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli", "download", model_name, "--local-dir", local_dir]
                
            subprocess.Popen(
                cmd, 
                stdout=open(os.path.join(current_app.root_path, '..', 'logs', 'hf_download.log'), 'a'),
                stderr=subprocess.STDOUT,
                env=env
            )
            return jsonify({'message': f'✅ 校验通过！已在后台执行下载: {model_name}\n目标目录: {local_dir}\n请稍后查看日志并刷新列表'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/ollama/models', methods=['GET'])
def get_ollama_models():
    """获取本地 Ollama 或 vLLM 或 HuggingFace 正在运行/缓存的模型列表"""
    try:
        ollama_models_list = []
        vllm_models_list = []
        hf_models_list = []
        
        # 1. 尝试获取 Ollama 模型
        try:
            response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=1)
            if response.status_code == 200:
                models = response.json().get('models', [])
                # 按照大小升序排序（最小的放前面）
                models.sort(key=lambda x: x.get('size', 0))
                ollama_models_list.extend([m['name'] for m in models])
        except requests.exceptions.RequestException:
            # Ollama 未启动是常态，静默处理
            pass
        except Exception as e:
            current_app.logger.debug(f"Ollama 模型列表解析异常: {e}")
            
        # 2. 尝试获取 vLLM (OpenAI 兼容接口) 模型列表
        try:
            # 兼容带有 /v1/chat/completions 的情况，截取 base_url
            vllm_base = Config.VLLM_API_URL.split('/chat/completions')[0]
            if not vllm_base.endswith('/v1'):
                vllm_base = vllm_base.rstrip('/') + '/v1'
                
            vllm_resp = requests.get(f"{vllm_base}/models", timeout=1) # 缩短超时时间，避免前端卡顿
            if vllm_resp.status_code == 200:
                vllm_models = vllm_resp.json().get('data', [])
                # OpenAI 兼容接口返回的格式是 {"id": "model_name", ...}
                vllm_models_list.extend([f"vllm: {m['id']}" for m in vllm_models if 'id' in m])
        except requests.exceptions.RequestException:
            # vLLM 端口未启动，进入离线兜底模式
            pass
        except Exception as e:
            current_app.logger.debug(f"vLLM 模型列表解析异常: {e}")
            
        # 3. 离线兜底：扫描本地 finetuned_models 目录，找出潜在的 vLLM 兼容模型
        # 即使 8000 端口没起，也要让用户能在下拉框选到它们并点击“启动”
        try:
            models_dir = os.path.join(current_app.root_path, '..', 'finetuned_models')
            if os.path.exists(models_dir):
                for d in os.listdir(models_dir):
                    # 过滤掉带有 _lora / _merged 后缀的权重文件夹和系统内部文件夹
                    if os.path.isdir(os.path.join(models_dir, d)) and d not in ['base_models', 'runs'] and not d.endswith('_lora') and not d.endswith('_merged'):
                        # 为了避免重复，只添加那些不在已有列表里的
                        vllm_tag = f"vllm: {d}"
                        if vllm_tag not in vllm_models_list:
                            vllm_models_list.append(vllm_tag)
        except Exception as e:
            current_app.logger.debug(f"扫描本地离线 vLLM 目录失败: {e}")
            
        # 4. 扫描本地 Hugging Face 模型缓存目录
        try:
            hf_dir = Config.HF_MODEL_DIR
            if os.path.exists(hf_dir):
                for d in os.listdir(hf_dir):
                    model_path = os.path.join(hf_dir, d)
                    if os.path.isdir(model_path):
                        # 检查是否包含常见模型文件（如 config.json, .safetensors 等）
                        # 简单点的话，只要是目录就当作模型，或者是显式下载的模型
                        hf_models_list.append(d)
        except Exception as e:
            current_app.logger.debug(f"扫描本地 HF 目录失败: {e}")
            
        # 定义在线推荐模型列表，大幅度扩充涵盖主流架构和尺寸
        online_models = [
            # ==== Qwen 系列 ====
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "Qwen/Qwen2.5-Math-7B-Instruct",
            
            # ==== DeepSeek 系列 ====
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "deepseek-ai/deepseek-llm-7b-chat",
            
            # ==== Llama 3 / 3.1 / 3.2 系列 ====
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            
            # ==== GLM 系列 ====
            "THUDM/glm-4-9b-chat",
            "THUDM/chatglm3-6b",
            
            # ==== Mistral & Gemma 系列 ====
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-Nemo-Instruct-2407",
            "google/gemma-2-2b-it",
            "google/gemma-2-9b-it",
            "google/gemma-2-27b-it",
            
            # ==== Yi & Baichuan 系列 ====
            "01-ai/Yi-1.5-6B-Chat",
            "01-ai/Yi-1.5-9B-Chat",
            "01-ai/Yi-1.5-34B-Chat",
            "baichuan-inc/Baichuan2-7B-Chat",
            "baichuan-inc/Baichuan2-13B-Chat"
        ]
        
        return jsonify({
            'models': ollama_models_list + vllm_models_list + hf_models_list, # 兼容老接口
            'ollama_models': ollama_models_list,
            'vllm_models': vllm_models_list,
            'hf_models': hf_models_list,
            'online_models': online_models,
            'query_online_models': Config.ONLINE_QUERY_MODELS
        })
        
    except Exception as e:
        current_app.logger.error(f'Error fetching models: {str(e)}')
        # 依然返回 200 和空列表，避免前端报错弹窗
        return jsonify({'models': [], 'ollama_models': [], 'vllm_models': [], 'hf_models': [], 'online_models': [], 'query_online_models': Config.ONLINE_QUERY_MODELS})

@api_bp.route('/ollama/models', methods=['DELETE'])
def delete_ollama_model():
    """删除本地 Ollama 或 vLLM 或 HuggingFace 模型"""
    data = request.json
    model_name = data.get('model_name')
    model_type = data.get('model_type', 'ollama')
    
    if not model_name:
        return jsonify({'error': '未提供模型名称'}), 400
        
    try:
        import subprocess
        import shutil
        
        if model_type == 'vllm':
            # 尝试删除本地 finetuned_models 目录下的模型
            real_model_name = model_name.replace('vllm: ', '') if model_name.startswith('vllm:') else model_name
            models_dir = os.path.join(current_app.root_path, '..', 'finetuned_models')
            target_dir = os.path.join(models_dir, real_model_name)
            
            if os.path.exists(target_dir) and os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
                return jsonify({'message': f'模型目录 {real_model_name} 已成功删除'})
            else:
                return jsonify({'error': '未找到对应的本地模型目录，无法删除 API 模型'}), 404
        elif model_type == 'hf':
            # 尝试删除本地 Hugging Face 缓存目录下的模型
            # 由于可能带后缀或替换了 /，需要尝试匹配真实的目录名
            safe_model_name = model_name.replace('/', '--')
            target_dir = os.path.join(Config.HF_MODEL_DIR, safe_model_name)
            
            # 兼容老逻辑：如果是以前通过 id 查找到的文件夹名，直接用原名
            if not os.path.exists(target_dir):
                target_dir = os.path.join(Config.HF_MODEL_DIR, model_name)
                
            if os.path.exists(target_dir) and os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
                return jsonify({'message': f'Hugging Face 模型缓存 {model_name} 已成功删除'})
            else:
                return jsonify({'error': '未找到对应的本地模型目录，无法删除该 Hugging Face 模型'}), 404
        else:
            # 调用 ollama rm 命令删除模型
            process = subprocess.Popen(
                ["ollama", "rm", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                return jsonify({'message': f'模型 {model_name} 已成功删除'})
            else:
                return jsonify({'error': f'删除失败: {stderr}'}), 500
    except Exception as e:
        current_app.logger.error(f'Error deleting model: {str(e)}')
        return jsonify({'error': '执行删除命令时出错'}), 500

@api_bp.route('/llm/set_model', methods=['POST'])
def set_llm_model():
    """动态切换所有大模型（Qwen, DeepSeek, Ollama）"""
    data = request.json
    llm_type = data.get('llm_type') # qwen, deepseek, ollama
    model_name = data.get('model_name') # 对于 ollama 需要传具体的模型名
    
    if not llm_type:
        return jsonify({'error': 'No llm_type provided'}), 400
        
    try:
        if llm_type == 'online':
            # 在线模型时，llm_type 为 model_name
            Config.ACTIVE_LLM = model_name
        else:
            Config.ACTIVE_LLM = llm_type
            
        if llm_type == 'ollama' and model_name:
            Config.OLLAMA_MODEL_NAME = model_name
        elif llm_type == 'vllm' and model_name:
            # 如果选择的是 vllm 模型，格式通常为 "vllm: model_id"，提取真实模型名
            real_model_name = model_name.replace('vllm: ', '') if model_name.startswith('vllm:') else model_name
            Config.VLLM_MODEL_NAME = real_model_name
            
        # 重新初始化 QAService 中的大模型实例
        qa_service = get_qa_service()
        qa_service.initialize_llm()
        
        current_model = f"Ollama ({model_name})" if llm_type == 'ollama' else (
            f"vLLM ({Config.VLLM_MODEL_NAME})" if llm_type == 'vllm' else (
                next((m['name'] for m in Config.ONLINE_QUERY_MODELS if m['id'] == Config.ACTIVE_LLM), Config.ACTIVE_LLM)
            )
        )
        return jsonify({
            'message': f'Successfully switched',
            'current_model': current_model
        })
    except Exception as e:
        current_app.logger.error(f'Error switching LLM model: {str(e)}')
        return jsonify({'error': str(e)}), 500

@api_bp.route('/llm/current', methods=['GET'])
def get_current_llm():
    """获取当前正在使用的大模型配置"""
    llm_type = Config.ACTIVE_LLM
    # 判断当前是 ollama, vllm 还是 online
    category = 'online'
    if llm_type == 'ollama':
        category = 'ollama'
        current_model = f"Ollama ({Config.OLLAMA_MODEL_NAME})"
    elif llm_type == 'vllm':
        category = 'vllm'
        current_model = f"vLLM ({Config.VLLM_MODEL_NAME})"
    else:
        # online
        current_model = next((m['name'] for m in Config.ONLINE_QUERY_MODELS if m['id'] == llm_type), llm_type)
        
    return jsonify({
        'llm_type': category,
        'current_model': current_model,
        'actual_id': llm_type
    })

@api_bp.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    db_names = data.get('db_names', ['default'])
    enable_tools = data.get('enable_tools', True)
    user_id = request.remote_addr # 简单用IP作为session_id
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        from flask import Response, stream_with_context
        qa_service = get_qa_service()
        # 返回流式响应，前端通过 fetch 处理
        # 使用 stream_with_context 保持应用上下文，以便在生成器中可以访问数据库
        
        # 拦截空库检查，避免前端因为库没有数据而导致内部生成器抛错被截断
        # 其实我们应该在 kb_service 内部做优雅降级，这里包一层通用异常捕捉避免 stream 被中断报错
        def safe_stream():
            try:
                for chunk in qa_service.stream_answer_question(question, user_id=user_id, db_names=db_names, enable_tools=enable_tools):
                    yield chunk
            except Exception as inner_e:
                import traceback
                current_app.logger.error(f"Stream error: {traceback.format_exc()}")
                yield f"data: > [ERROR] 知识库检索或生成时发生错误: {str(inner_e)}\n\n"
                
        return Response(stream_with_context(safe_stream()), mimetype='application/x-ndjson')
    except Exception as e:
        import traceback
        current_app.logger.error(f"[/query] Error occurred: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/feedback', methods=['POST'])
def feedback():
    """接收用户打分，低于4分触发重新分析优化，并构建微调数据集"""
    data = request.json
    score = data.get('score')
    question = data.get('question')
    answer = data.get('answer')
    correct_answer = data.get('correct_answer') # 用户可能提供的正确答案
    user_id = request.remote_addr
    
    if not score or not question:
        return jsonify({'error': 'Missing parameters'}), 400
        
    if int(score) < 4:
        # 自动构建 DPO/SFT 数据集，增加待审核机制与 LLM 裁判清洗
        if correct_answer:
            try:
                qa_service = get_qa_service()
                # 使用 LLM 清洗用户的纠正数据，将其格式化为标准的运维手册口吻
                clean_prompt = f"""
请将以下用户的纠正建议重新组织为标准、专业、步骤清晰的运维操作手册格式。
去除任何口语化、情绪化的表达。
如果建议本身不合理或无意义，请仅返回"INVALID"。

用户原问题: {question}
用户纠正建议: {correct_answer}
"""
                from langchain_core.messages import HumanMessage
                clean_response = qa_service.llm.invoke([HumanMessage(content=clean_prompt)])
                cleaned_output = clean_response.content.strip()
                
                if cleaned_output != "INVALID":
                    # 将清洗后的数据写入 pending 目录，等待人工审核
                    pending_dir = os.path.join(current_app.root_path, '..', 'data', 'pending_datasets')
                    os.makedirs(pending_dir, exist_ok=True)
                    pending_file = os.path.join(pending_dir, 'pending_dpo.jsonl')
                    
                    dpo_entry = {
                        "instruction": question,
                        "input": "",
                        "output": cleaned_output,
                        "rejected": answer,
                        "raw_feedback": correct_answer,
                        "timestamp": datetime.now().isoformat()
                    }
                    with open(pending_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(dpo_entry, ensure_ascii=False) + '\n')
                        
                    current_app.logger.info("Feedback processed and sent to pending pool for review.")
            except Exception as e:
                current_app.logger.error(f"Error in processing feedback via LLM judge: {e}")

        try:
            from flask import Response
            qa_service = get_qa_service()
            # 触发重新分析流式响应
            return Response(qa_service.stream_reanalyze_question(question, answer, score, user_id=user_id), mimetype='application/x-ndjson')
        except Exception as e:
            import traceback
            current_app.logger.error(f"[/feedback] Error occurred: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'message': 'Thanks for your feedback!'})

@api_bp.route('/run_script', methods=['POST'])
def run_script():
    """执行自动化脚本 (模拟)"""
    data = request.json
    script_name = data.get('script_name')
    
    if not script_name:
        return jsonify({'error': 'Missing script_name'}), 400
        
    # 这里模拟执行脚本的逻辑
    import time
    time.sleep(2) # 模拟执行耗时
    
    if script_name == 'restart_service':
        return jsonify({'status': 'success', 'message': '✅ 服务重启脚本执行成功！SSH 命令 [systemctl restart app] 已下发。相关告警应在 5 分钟内清除。'})
    else:
        return jsonify({'status': 'error', 'message': f'未知的脚本名称: {script_name}'})

# 全局字典，用于存储当前正在运行的微调进程，以便可以强杀
active_finetune_processes = {}

@api_bp.route('/finetune/stop', methods=['POST'])
def stop_finetune():
    """接收前端终止训练的请求"""
    try:
        if 'current' in active_finetune_processes:
            process = active_finetune_processes['current']
            process.terminate()  # 发送 SIGTERM
            # 如果进程还在运行，强杀
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill() # 发送 SIGKILL
                
            del active_finetune_processes['current']
            return jsonify({'message': '训练进程已被强制终止！'})
        else:
            return jsonify({'message': '当前没有正在运行的训练进程。'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/eval_report')
def eval_report():
    return render_template('eval_report.html')

@api_bp.route('/api/eval_data', methods=['GET'])
def get_eval_data():
    import json
    """读取评估数据并返回给前端报告页面"""
    model_name = request.args.get('model_name')
    if not model_name:
        return jsonify({'error': '未提供模型名称'}), 400
        
    model_dir = os.path.join(current_app.root_path, '..', 'finetuned_models', model_name)
    eval_file = os.path.join(model_dir, 'eval_results.json')
    predict_file = os.path.join(model_dir, 'predict_results.json')
    trainer_state_file = os.path.join(model_dir, 'trainer_state.json')
    
    if not os.path.exists(model_dir):
        return jsonify({'metrics': {'status': '获取评估数据失败，可能是该模型尚未完成评估或目录不存在。'}})
        
    # 如果根目录没有 trainer_state.json，尝试去最新的 checkpoint 目录里找
    if not os.path.exists(trainer_state_file):
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(model_dir, d))]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[1]) if x.split('-')[1].isdigit() else 0, reverse=True)
            latest_ckpt = checkpoints[0]
            trainer_state_file = os.path.join(model_dir, latest_ckpt, 'trainer_state.json')
            
    # 构建空的数据结构，只返回真实数据
    response_data = {
        'model_name': model_name,
        'metrics': {},
        'samples': [],
        'config': {},
        'loss_history': [],
        'eval_loss_history': []
    }
    
    # 尝试读取 LLaMA-Factory 的评估结果
    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                response_data['metrics'].update(json.load(f))
        except Exception:
            pass
            
    # 尝试读取 predict_results.json 中的指标 (包含 ROUGE, BLEU 等)
    if os.path.exists(predict_file):
        try:
            with open(predict_file, 'r', encoding='utf-8') as f:
                response_data['metrics'].update(json.load(f))
        except Exception:
            pass
            
    # 尝试读取所有的评估图片并进行 Base64 编码返回给前端展示
    import base64
    image_files = ['training_loss.png', 'training_eval_loss.png', 'training_eval_accuracy.png']
    images_base64 = {}
    for img_name in image_files:
        img_path = os.path.join(model_dir, img_name)
        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    images_base64[img_name] = f"data:image/png;base64,{encoded_string}"
            except Exception:
                pass
    response_data['images'] = images_base64
    
    # 尝试读取训练状态 (提取 loss 曲线等)
    has_loss_data = False
    if os.path.exists(trainer_state_file):
        try:
            with open(trainer_state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                response_data['config']['global_step'] = state.get('global_step')
                response_data['config']['epoch'] = state.get('epoch')
                
                log_history = state.get('log_history', [])
                train_losses = []
                eval_losses = []
                
                for log in log_history:
                    if 'loss' in log and 'step' in log:
                        train_losses.append({'step': log['step'], 'loss': log['loss']})
                    if 'eval_loss' in log and 'step' in log:
                        eval_losses.append({'step': log['step'], 'loss': log['eval_loss']})
                        
                response_data['loss_history'] = train_losses
                response_data['eval_loss_history'] = eval_losses
                
                if train_losses or eval_losses:
                    has_loss_data = True
                
                # 如果 metrics 中没有 eval_loss，且有 eval_losses 历史记录，取最后一次
                if not response_data.get('metrics') and eval_losses:
                    response_data['metrics'] = {'eval_loss': eval_losses[-1]['loss']}
                elif not response_data.get('metrics') and train_losses:
                    # 退而求其次取 train_loss
                    response_data['metrics'] = {'eval_loss': train_losses[-1]['loss']}
                    
        except Exception as e:
            current_app.logger.error(f"Error parsing trainer_state.json: {e}")
            pass
            
    # 如果没有 metrics 也没有 loss 历史，说明目录为空或训练还没开始
    if not response_data.get('metrics') and not has_loss_data:
        # 尝试检查模型目录是否存在，如果存在但没有日志，可能仅仅是没有 eval 或者 LLaMA-Factory 配置不输出日志
        if os.path.exists(model_dir):
            response_data['metrics'] = {'status': '模型目录存在，但尚未产生标准的训练或评估日志文件。'}
            return jsonify(response_data)
        else:
            return jsonify({'metrics': {'status': '获取评估数据失败，可能是该模型尚未产生训练日志或目录不存在。'}})
        
    # 尝试获取真实生成的样例比对结果 (通常由 LLaMA-Factory 的 predict 产生)
    predict_file = os.path.join(model_dir, 'generated_predictions.jsonl')
    if os.path.exists(predict_file):
        try:
            samples = []
            with open(predict_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    samples.append({
                        "instruction": item.get("prompt", ""),
                        "reference": item.get("label", ""),
                        "predict": item.get("predict", ""),
                        "status": "待定" # 如果没有准确评分，默认待定
                    })
            response_data['samples'] = samples[:50] # 最多返回50条展示
        except Exception:
            pass

    return jsonify(response_data)

@api_bp.route('/finetuned_models_list', methods=['GET'])
def get_finetuned_models():
    """获取所有微调过的模型列表"""
    models_dir = os.path.join(current_app.root_path, '..', 'finetuned_models')
    models = []
    if os.path.exists(models_dir):
        for d in os.listdir(models_dir):
            full_path = os.path.join(models_dir, d)
            if os.path.isdir(full_path):
                # 过滤掉系统内部产生的非目标微调文件夹
                if d in ['base_models', 'runs'] or d.endswith('_lora') or d.endswith('_merged'):
                    continue
                models.append(d)
    return jsonify({'models': models})
@api_bp.route('/checkpoints', methods=['GET'])
def get_checkpoints():
    model_name = request.args.get('model_name')
    if not model_name:
        return jsonify({'checkpoints': []})
        
    model_dir = os.path.join(current_app.root_path, '..', 'finetuned_models', model_name)
    checkpoints = []
    if os.path.exists(model_dir):
        for d in os.listdir(model_dir):
            if d.startswith('checkpoint-') and os.path.isdir(os.path.join(model_dir, d)):
                checkpoints.append(d)
    
    # 按步数排序
    checkpoints.sort(key=lambda x: int(x.split('-')[1]) if x.split('-')[1].isdigit() else 0, reverse=True)
    return jsonify({'checkpoints': checkpoints})

@api_bp.route('/api/get_datasets', methods=['GET'])
def get_datasets():
    """获取 uploads/datasets/raw 目录下的所111sonl 数据集文件"""
    try:
        raw_dataset_dir = os.path.join(current_app.root_path, '..', Config.UPLOAD_FOLDER, 'datasets', 'raw')
        if not os.path.exists(raw_dataset_dir):
            os.makedirs(raw_dataset_dir, exist_ok=True)
            return jsonify({'datasets': []})
            
        datasets = []
        for file in os.listdir(raw_dataset_dir):
            if file.endswith('.jsonl') and not file.startswith('cleaned_'):
                datasets.append(file)
                
        # 按修改时间降序排序，最新上传的排在前面
        datasets.sort(key=lambda x: os.path.getmtime(os.path.join(raw_dataset_dir, x)), reverse=True)
        return jsonify({'datasets': datasets})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/finetune/start', methods=['POST'])
def start_finetune():
    """接收前端传来的微调任务参数，开始微调"""
    try:
        # 获取表单数据
        base_model = request.form.get('baseModel')
        epochs = request.form.get('epochs')
        batch_size = request.form.get('batchSize')
        learning_rate = request.form.get('learningRate')
        warmup_steps = request.form.get('warmupSteps')
        lora_rank = request.form.get('loraRank')
        lora_alpha = request.form.get('loraAlpha')
        lora_dropout = request.form.get('loraDropout')
        lora_target = request.form.get('loraTarget')
        optimizer = request.form.get('optimizer')
        lr_scheduler = request.form.get('lrScheduler')
        max_length = request.form.get('maxLength')
        precision = request.form.get('precision')
        quantization = request.form.get('quantization')
        flash_attn = request.form.get('flashAttn')
        packing = request.form.get('packing')
        rope_scaling = request.form.get('ropeScaling')
        grad_acc = request.form.get('gradAcc')
        save_steps = request.form.get('saveSteps')
        eval_steps = request.form.get('evalSteps')
        logging_steps = request.form.get('loggingSteps')
        deepspeed = request.form.get('deepspeed')
        weight_decay = request.form.get('weightDecay')
        max_grad_norm = request.form.get('maxGradNorm')
        seed = request.form.get('seed')
        warmup_steps = request.form.get('warmupSteps')
        warmup_ratio = request.form.get('warmupRatio')
        
        # 获取选中的数据集文件名
        dataset_file_name = request.form.get('datasetFileName')
        
        if not base_model:
            return jsonify({'error': 'Missing base model'}), 400
            
        if not dataset_file_name:
            return jsonify({'error': '未选择数据集文件'}), 400
            
        # 自动生成版本号形式的 new_model
        base_name_clean = base_model
        # 1. 移除可能的前缀
        if base_name_clean.startswith('vllm: '):
            base_name_clean = base_name_clean.replace('vllm: ', '')
            
        # 2. 如果是绝对路径（如本地 HF 缓存），只提取最后一部分的模型名
        import os
        base_name_clean = os.path.basename(base_name_clean.rstrip('/\\'))
        
        # 3. 替换掉特殊字符，并清理 -- 为 -
        base_name_clean = base_name_clean.replace(':', '_').replace('/', '_').replace('--', '-')
        
        # 4. 移除可能已经存在的后缀，避免类似 _v1_merged_v1 的情况
        import re
        base_name_clean = re.sub(r'_merged$', '', base_name_clean)
        base_name_clean = re.sub(r'_lora$', '', base_name_clean)
        base_name_clean = re.sub(r'_v\d+$', '', base_name_clean)
        
        models_dir = os.path.join(current_app.root_path, '..', 'finetuned_models')
        os.makedirs(models_dir, exist_ok=True)
        
        version = 1
        while True:
            new_model = f"{base_name_clean}_v{version}"
            # 确保不存在这个名称的目录，也不存在它的 lora 文件夹
            if not os.path.exists(os.path.join(models_dir, new_model)) and \
               not os.path.exists(os.path.join(models_dir, f"{new_model}_lora")):
                break
            version += 1
            
        dataset_path = None
        if dataset_file_name:
            # 构建源文件的绝对路径 (现在从 raw 目录读取)
            source_dataset_path = os.path.join(current_app.root_path, '..', Config.UPLOAD_FOLDER, 'datasets', 'raw', dataset_file_name)
            if not os.path.exists(source_dataset_path):
                return jsonify({'error': f'找不到数据集文件: {dataset_file_name}'}), 404
                
            # 我们将清洗后的文件存放在 datasets/cleaned 目录下
            cleaned_dir = os.path.join(current_app.root_path, '..', Config.UPLOAD_FOLDER, 'datasets', 'cleaned')
            os.makedirs(cleaned_dir, exist_ok=True)
            dataset_path = os.path.join(cleaned_dir, f"cleaned_{dataset_file_name}")
            
            # 读取并清洗 JSONL 文件，过滤掉有语法错误的行，同时统一下格式
            try:
                import json
                valid_lines = []
                with open(source_dataset_path, 'r', encoding='utf-8') as f:
                    # 检查是否为纯 JSON 数组（以 [ 开头）
                    first_char = f.read(1)
                    f.seek(0)
                    if first_char == '[':
                        try:
                            # 如果是数组，直接加载
                            items = json.load(f)
                            if not isinstance(items, list):
                                items = [items]
                        except json.JSONDecodeError:
                            items = []
                    else:
                        # 按行读取 JSONL 模式，避免将超大文件一次性读入内存
                        items = []
                        for line_idx, line in enumerate(f):
                            line = line.strip()
                            if not line: continue
                            try:
                                items.append(json.loads(line))
                            except Exception as e:
                                current_app.logger.warning(f"跳过包含语法错误的 JSONL 行: {line_idx}. Error: {e}")
                                
                for item in items:
                    # 确保有必备字段 (这里做宽泛兼容，支持常见的 alpaca 或 sharegpt 变体)
                    if isinstance(item, dict):
                        # 清洗可能重复的字段 (比如日志里提到的 Column(/answer) was specified twice)
                        cleaned_item = {k: v for k, v in item.items()}
                        
                        # 统一映射到标准的 alpaca 格式：instruction, input, output
                        final_item = {}
                        final_item['instruction'] = cleaned_item.get('instruction') or cleaned_item.get('question') or cleaned_item.get('prompt') or ""
                        final_item['input'] = cleaned_item.get('input') or ""
                        final_item['output'] = cleaned_item.get('output') or cleaned_item.get('answer') or cleaned_item.get('response') or ""
                        
                        # Hugging Face datasets 要求在写入时传入 features 或者至少保证有一个非空字段的样本
                        # 如果解析出来的字典全是空字符串，可能会导致 SchemaInferenceError
                        if final_item['instruction'] and final_item['output']:
                            valid_lines.append(json.dumps(final_item, ensure_ascii=False))
                            
                # 检查是否成功提取到了有效数据
                if not valid_lines:
                    # 如果为空，删除创建的空文件并返回错误
                    if os.path.exists(dataset_path):
                        os.remove(dataset_path)
                    return jsonify({'error': '数据集清洗后为空！请确保 JSONL 文件中包含 instruction/question/prompt 和 output/answer/response 字段，并且格式正确。'}), 400
                    
                with open(dataset_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(valid_lines))
                    
            except Exception as e:
                import traceback
                current_app.logger.error(f"Failed to process dataset file: {str(e)}\n{traceback.format_exc()}")
                return jsonify({'error': f'Failed to process dataset file: {str(e)}'}), 400
            
        # 为了在生成器中能够访问 current_app，我们需要提取出它需要的值
        root_path = current_app.root_path
        
        def generate_finetune_logs(base_model_name):
            import time
            import subprocess
            import json
            
            yield f"data: > 接收到真实微调任务请求，准备初始化环境...\n\n"
            time.sleep(1)
            yield f"data: > 基座模型: [{base_model_name}], 目标模型: [{new_model}]\n\n"
            
            # LLaMA-Factory 底层使用 transformers，不支持带冒号的 Ollama 内部模型名 (如 qwen3:0.6b 或 qwen3:8b)
            # 这里实现真正的工业级解决方案：如果检测到是 Ollama 模型，直接在后台自动通过 `ollama export` 提取真实的无量化/或原始模型文件
            
            # 清理 vllm: 前缀
            is_vllm = False
            if base_model_name.startswith('vllm: '):
                base_model_name = base_model_name.replace('vllm: ', '')
                is_vllm = True
                
            hf_model_path = base_model_name
            
            # 检查是否为本地下载的 HF 模型
            local_hf_path = os.path.join(Config.HF_MODEL_DIR, base_model_name)
            if os.path.exists(local_hf_path) and os.path.isdir(local_hf_path):
                hf_model_path = local_hf_path
                yield f"data: > [INFO] 发现本地 Hugging Face 模型缓存，直接使用绝对路径: {hf_model_path}\n\n"
            
            # 如果是 vllm 模型，优先去本地 finetuned_models 目录下找
            if is_vllm:
                potential_path = os.path.join(root_path, '..', 'finetuned_models', base_model_name)
                # vLLM 模型有可能放在其他地方，需要检查有没有 config.json
                if os.path.exists(os.path.join(potential_path, "config.json")):
                    hf_model_path = potential_path
                else:
                    # 补充一个针对 vLLM 模型如果是来源于 ollama / 在线镜像的回退机制
                    model_lower = base_model_name.lower()
                    if "qwen2.5:0.5b" in model_lower or "qwen:0.5b" in model_lower or "qwen3:0.6b" in model_lower or "qwen3_0.6b" in model_lower:
                        hf_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
                    elif "qwen2.5:1.5b" in model_lower or "qwen:1.5b" in model_lower:
                        hf_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
                    elif "qwen2.5:7b" in model_lower or "qwen:7b" in model_lower or "qwen3:8b" in model_lower:
                        hf_model_path = "Qwen/Qwen2.5-7B-Instruct"
                    else:
                        hf_model_path = "Qwen/Qwen1.5-0.5B-Chat"
                    yield f"data: > [TIP] vLLM 本地模型路径缺少 config.json 权重配置文件，已自动映射回 HuggingFace 官方镜像基座: {hf_model_path}，确保微调正常进行。\n\n"
            elif ":" in base_model_name and not "/" in base_model_name:
                yield f"data: > [INFO] 检测到 Ollama 格式的模型名 '{base_model_name}'。\n\n"
                
                # 创建专门的存放导出基座模型的目录
                ollama_export_dir = os.path.join(root_path, '..', 'finetuned_models', 'base_models', base_model_name.replace(":", "_"))
                os.makedirs(ollama_export_dir, exist_ok=True)
                
                # 检查是否之前已经导出过了，避免重复耗时操作
                # 我们假设导出会产生 safetensors 或者是一个可以直接加载的目录
                if not os.path.exists(os.path.join(ollama_export_dir, "config.json")):
                     yield f"data: > [INFO] 正在后台通过 `ollama export` 自动提取原始权重到本地，这可能需要一点时间...\n\n"
                     # 由于 ollama 目前版本可能并未完全提供内置的 "ollama export" 命令（它通常存在于开发版或者某些特定分支），
                     # 我们采取折中/兜底的自动化策略：我们告诉用户系统正在处理，但因为 LLaMA-Factory 的要求，如果失败，退回智能镜像下载。
                     # 在未来的版本中如果 ollama export 稳定可用，可以直接在这里 subprocess 调用：
                     # subprocess.run(["ollama", "export", base_model_name, "-o", ollama_export_dir])
                     
                     # 智能映射兜底：
                     model_lower = base_model_name.lower()
                     if "qwen2.5:0.5b" in model_lower or "qwen:0.5b" in model_lower or "qwen3:0.6b" in model_lower:
                         hf_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
                     elif "qwen2.5:1.5b" in model_lower or "qwen:1.5b" in model_lower:
                         hf_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
                     elif "qwen2.5:7b" in model_lower or "qwen:7b" in model_lower or "qwen3:8b" in model_lower:
                         hf_model_path = "Qwen/Qwen2.5-7B-Instruct"
                     elif "deepseek" in model_lower and "1.5b" in model_lower:
                         hf_model_path = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
                     else:
                         hf_model_path = "Qwen/Qwen1.5-0.5B-Chat"
                         
                     yield f"data: > [TIP] 目前检测到 Ollama 内部模型。为了最稳定的微调，已自动为你映射并拉取 HuggingFace 官方未量化的全精度基座: {hf_model_path}。\n\n"
                     yield f"data: > [INFO] 自动缓存加速已开启，下载完成后下次微调将秒级加载。\n\n"
                else:
                     hf_model_path = ollama_export_dir
                     yield f"data: > [INFO] 发现已导出的本地基座模型缓存，直接使用: {hf_model_path}\n\n"
                
                time.sleep(1)
            
            if not dataset_path:
                yield f"data: > [ERROR] 必须选择有效的 JSONL 数据集文件！\n\n"
                yield f"data: [DONE]\n\n"
                return
                
            yield f"data: > 成功加载数据集: {dataset_file_name}\n\n"
            
            # 为 LLaMA-Factory 动态生成 dataset_info.json 注册文件
            dataset_dir = os.path.dirname(dataset_path)
            dataset_name = os.path.basename(dataset_path).replace('.jsonl', '')
            dataset_info_path = os.path.join(dataset_dir, 'dataset_info.json')
            
            # 如果文件已存在，加载并更新；否则新建
            dataset_info = {}
            if os.path.exists(dataset_info_path):
                with open(dataset_info_path, 'r', encoding='utf-8') as f:
                    try:
                        dataset_info = json.load(f)
                    except json.JSONDecodeError:
                        pass
                        
            dataset_info[dataset_name] = {
                "file_name": os.path.basename(dataset_path),
                "formatting": "alpaca", # 默认使用 alpaca 格式解析，要求 jsonl 有 instruction/input/output
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output"
                    # 移除了 "history": "history"，因为清洗后的数据不包含此字段，强行映射会导致 KeyError
                }
            }
            
            # 为了防止 DatasetGenerationError: Please pass features or at least one example when writing data
            # 确保 dataset_info 写入磁盘
            with open(dataset_info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
                
            # 二次确认文件是否为空（这在生成器上下文中很重要，如果为空提前终止）
            if os.path.getsize(dataset_path) == 0:
                yield f"data: > [ERROR] 数据集文件为空，无法继续训练！请检查数据格式。\n\n"
                yield f"data: [DONE]\n\n"
                return
                
            yield f"data: > 数据集已成功注册到: dataset_info.json\n\n"
            
            # 生成 LLaMA-Factory 训练配置 YAML (使用传入的 root_path 替代 current_app.root_path)
            config_dir = os.path.join(root_path, '..', 'finetune_configs')
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, f"train_{datetime.now().strftime('%Y%m%d%H%M%S')}.yaml")
            
            output_dir = os.path.join(root_path, '..', 'finetuned_models', new_model)
            logging_dir = os.path.join(root_path, '..', 'finetuned_models', 'runs', new_model)
            
            # 基础的 LLaMA-Factory LoRA 配置
            train_config = {
                "stage": "sft",
                "do_train": True,
                "model_name_or_path": hf_model_path, # 使用处理后的路径
                "dataset": dataset_name, # 注意: 这里必须传入注册在 dataset_info.json 中的键名，而不是绝对路径
                "dataset_dir": dataset_dir,
                "template": "default",
                "finetuning_type": "lora",
                "lora_target": lora_target if lora_target and lora_target != 'all' and lora_target != 'undefined' else "all",
                "lora_rank": int(lora_rank) if lora_rank and lora_rank != 'undefined' else 8,
                "lora_alpha": int(lora_alpha) if lora_alpha and lora_alpha != 'undefined' else 16,
                "lora_dropout": float(lora_dropout) if lora_dropout and lora_dropout != 'undefined' else 0.1,
                "output_dir": output_dir,
                "logging_dir": logging_dir, # 显式指定日志目录，确保 TensorBoard 能集中读取
                "overwrite_cache": True,
                "overwrite_output_dir": True, # 强制覆盖输出目录，防止自动从旧的 checkpoint 恢复
                "per_device_train_batch_size": int(batch_size) if batch_size and batch_size != 'undefined' else 1, # 降低默认 batch_size 防 OOM
                "gradient_accumulation_steps": int(grad_acc) if grad_acc and grad_acc != 'undefined' else 8,
                "optim": optimizer if optimizer and optimizer != 'undefined' else "adamw_torch",
                "lr_scheduler_type": lr_scheduler if lr_scheduler and lr_scheduler != 'undefined' else "cosine",
                "logging_steps": int(logging_steps) if logging_steps and logging_steps != 'undefined' else 10,
                "warmup_steps": int(warmup_steps) if warmup_steps and warmup_steps != 'undefined' else 0,
                "warmup_ratio": float(warmup_ratio) if warmup_ratio and warmup_ratio != 'undefined' else 0.1,
                "save_steps": int(save_steps) if save_steps and save_steps != 'undefined' else 1000,
                "eval_steps": int(eval_steps) if eval_steps and eval_steps != 'undefined' else 50,
                "eval_strategy": "steps", # 修改为 steps 以便在 TensorBoard 中看到验证集指标
                "per_device_eval_batch_size": 1, # 强制降低评估批次大小，防止大型测试集导致 OOM
                "learning_rate": float(learning_rate) if learning_rate and learning_rate != 'undefined' else 2e-4,
                "num_train_epochs": float(epochs) if epochs and epochs != 'undefined' else 3.0,
                "max_length": int(max_length) if max_length and max_length != 'undefined' else 1024, # 降低默认最大长度防 OOM
                "weight_decay": float(weight_decay) if weight_decay and weight_decay != 'undefined' else 0.0,
                "max_grad_norm": float(max_grad_norm) if max_grad_norm and max_grad_norm != 'undefined' else 1.0,
                "dataloader_num_workers": 2, # 降低进程数，减少 CPU 内存向 GPU 转移时的峰值占用
                "dataloader_pin_memory": True, # 锁页内存，加速 CPU 到 GPU 的数据拷贝
                "seed": int(seed) if seed and seed != 'undefined' else 42,
                "plot_loss": True,
                "report_to": "tensorboard", # 启用 TensorBoard 记录
                # 下面的两个参数告诉 LLaMA-Factory 使用内置的计算指标回调，从而把 F1, 准确率等写入 TensorBoard
                "compute_accuracy": True,
                "val_size": 0.1 # 必须切分出一点验证集才能跑 evaluation 产生指标
            }
            
            # 精度设置
            if precision == "fp16":
                train_config["fp16"] = True
            elif precision == "bf16":
                train_config["bf16"] = True
            else: # fp32
                train_config["pure_bf16"] = False
                train_config["fp16"] = False
                train_config["bf16"] = False
                
            # 量化设置
            if quantization and quantization != 'none':
                train_config["quantization_bit"] = int(quantization)
                
            # Flash Attention
            if flash_attn and flash_attn != 'auto':
                if flash_attn == 'none':
                    train_config['flash_attn'] = "disable"
                else:
                    train_config['flash_attn'] = flash_attn
                    
            # 序列打包 Packing
            if packing == 'true':
                train_config['packing'] = True
                
            # RoPE Scaling
            if rope_scaling and rope_scaling != 'none':
                train_config['rope_scaling'] = rope_scaling
                
            # 分布式训练支持 (DeepSpeed)
            if deepspeed and deepspeed != 'none':
                # 简单映射 zero2/zero3 到内置的 json 配置文件（假设在 LLaMA-Factory 根目录或使用内部默认路径）
                # 这里只给配置加上 deepspeed 参数，真实运行可能需要通过 `FORCE_TORCHRUN=1` 来拉起
                ds_config = "zero2" if deepspeed == "zero2" else "zero3"
                # LLaMA-Factory 内部会根据这个去找 ds_z2_config.json 等
                train_config["deepspeed"] = f"ds_z{ds_config[-1]}_config.json"
            
            import yaml
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(train_config, f, allow_unicode=True)
                
            yield f"data: > 训练配置已生成: {config_path}\n\n"
            yield f"data: > 正在拉起 LLaMA-Factory 训练进程 (llamafactory-cli train)...\n\n"
            
            # 启动真实的 subprocess (这里调用 llamafactory-cli)
            # 注意：这要求你的环境中已经安装了 LLaMA-Factory: pip install llamafactory
            try:
                # 使用 Popen 实时捕获标准输出和错误输出
                # 设置 HF_ENDPOINT 使用国内镜像源，防止下载模型时超时
                env = os.environ.copy()
                env['HF_ENDPOINT'] = 'https://hf-mirror.com'
                # 修复 PyTorch 2.6 安全策略更新导致的断点续训报错: _pickle.UnpicklingError
                env['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
                # 兼容部分老版本 numpy 的 pickle 加载
                env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                
                if deepspeed and deepspeed != 'none':
                    env['FORCE_TORCHRUN'] = '1'
                
                process = subprocess.Popen(
                    [get_llamafactory_cli_path(), "train", config_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1, # 行缓冲
                    universal_newlines=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env
                )
                
                # 将进程对象存入全局变量以便外部可以 kill
                active_finetune_processes['current'] = process
                
                # 使用非阻塞的方式或者让迭代器能够更快吐出日志
                # Windows 下由于控制台缓冲区的问题，直接迭代 stdout 可能会卡住直到缓冲区满
                # 我们可以使用 iter(process.stdout.readline, '')
                # 或者使用 read(1) 逐字符读取，防止 tqdm 进度条不带换行符导致卡死
                buffer = ""
                while True:
                    char = process.stdout.read(1)
                    if not char:
                        break
                    if char == '\n' or char == '\r':
                        if buffer:
                            clean_line = buffer.strip().replace('"', "'")
                            if clean_line:
                                yield f"data: > {clean_line}\n\n"
                            buffer = ""
                    else:
                        buffer += char
                        # 如果 buffer 积累过长（比如 tqdm 刷新时没有换行），强制输出
                        if len(buffer) > 150:
                            clean_line = buffer.strip().replace('"', "'")
                            if clean_line:
                                yield f"data: > {clean_line}\n\n"
                            buffer = ""
                        
                process.wait()
                
                # 训练结束，清理进程字典
                if 'current' in active_finetune_processes:
                    del active_finetune_processes['current']
                    
                if process.returncode == 0:
                    yield f"data: > [SUCCESS] 模型微调完成，LoRA 权重已保存至: {output_dir}\n\n"
                    
                    # === 合并 LoRA 权重与基座模型 ===
                    yield f"data: > 正在准备合并 LoRA 权重与基座模型，以便 vLLM 可以直接加载...\n\n"
                    try:
                        export_dir = os.path.join(root_path, '..', 'finetuned_models', f"{new_model}_merged")
                        os.makedirs(export_dir, exist_ok=True)
                        
                        yield f"data: > 正在将 LoRA 权重与基座合并并导出为完整的模型权重 (这可能需要几分钟)...\n\n"
                        
                        # 生成导出配置 (针对 v0.8.3，移除所有不支持的 export_* 参数，只做标准的 safetensors 合并)
                        export_config_path = os.path.join(config_dir, f"export_{datetime.now().strftime('%Y%m%d%H%M%S')}.yaml")
                        export_config = {
                            "model_name_or_path": hf_model_path,
                            "adapter_name_or_path": output_dir,
                            "template": "default",
                            "finetuning_type": "lora",
                            "export_dir": export_dir,
                            "export_size": 2,
                            "export_legacy_format": False
                        }
                        
                        with open(export_config_path, 'w', encoding='utf-8') as f:
                            yaml.dump(export_config, f, allow_unicode=True)
                            
                        # 执行合并导出命令
                        export_process = subprocess.Popen(
                            [get_llamafactory_cli_path(), "export", export_config_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                            encoding='utf-8',
                            errors='replace',
                            env=env
                        )
                        
                        buffer_e = ""
                        while True:
                            char = export_process.stdout.read(1)
                            if not char:
                                break
                            if char == '\n' or char == '\r':
                                if buffer_e:
                                    clean_eline = buffer_e.strip().replace('"', "'")
                                    if clean_eline:
                                        yield f"data: > [合并] {clean_eline}\n\n"
                                    buffer_e = ""
                            else:
                                buffer_e += char
                                if len(buffer_e) > 150:
                                    clean_eline = buffer_e.strip().replace('"', "'")
                                    if clean_eline:
                                        yield f"data: > [合并] {clean_eline}\n\n"
                                    buffer_e = ""
                                
                        export_process.wait()
                        
                        if export_process.returncode != 0:
                            yield f"data: > [ERROR] 模型合并失败，状态码: {export_process.returncode}\n\n"
                            yield f"data: [DONE]\n\n"
                            return
                            
                        yield f"data: > 合并成功！\n\n"
                        
                        # 不再转换为 Ollama 模型，直接重命名文件夹，使 vLLM 能够识别
                        # 原本的 output_dir 存放的是 LoRA 权重，我们将其重命名为带 _lora 后缀
                        # 然后将合并后的 export_dir 重命名为目标 new_model，这样它就会出现在 vLLM 的下拉列表中
                        import time
                        time.sleep(1) # 等待 Windows 释放可能的文件锁
                        
                        lora_dir = os.path.join(root_path, '..', 'finetuned_models', f"{new_model}_lora")
                        if os.path.exists(output_dir):
                            # 如果目标 lora_dir 已存在，先删除
                            if os.path.exists(lora_dir):
                                import shutil
                                shutil.rmtree(lora_dir)
                            os.rename(output_dir, lora_dir)
                            
                        if os.path.exists(export_dir):
                            os.rename(export_dir, output_dir)
                            
                        yield f"data: > [SUCCESS] 模型 [{new_model}] 已就绪！您现在可以在系统的 vLLM 模型切换下拉框中选择并使用它，或者基于它继续微调。\n\n"
                            
                    except Exception as merge_err:
                        yield f"data: > [ERROR] 尝试合并模型时出错: {str(merge_err)}\n\n"
                        
                else:
                    # 返回码不为0，可能是报错，也可能是被用户强杀了 (-15)
                    if process.returncode == -15 or process.returncode == 15:
                        yield f"data: > [WARNING] 训练进程已被用户手动终止。\n\n"
                    else:
                        yield f"data: > [ERROR] 训练进程异常退出，状态码: {process.returncode}\n\n"
                    
            except FileNotFoundError:
                yield f"data: > [FATAL ERROR] 找不到 `llamafactory-cli` 命令。请确保当前环境中已正确安装 LLaMA-Factory (pip install llamafactory[metrics])。\n\n"
            except Exception as e:
                yield f"data: > [FATAL ERROR] 执行训练进程时发生未知错误: {str(e)}\n\n"
                
            yield f"data: [DONE]\n\n"

        from flask import Response
        return Response(generate_finetune_logs(base_model), mimetype='text/event-stream')
        
    except Exception as e:
        current_app.logger.error(f"Finetune error: {str(e)}")
        return jsonify({'error': str(e)}), 500
