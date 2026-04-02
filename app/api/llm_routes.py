"""
LLM 模型管理相关路由：启动服务、下载模型、查询、反馈
"""
from flask import request, jsonify, Response, current_app
from app.config import Config
from app.api.common import api_bp, get_qa_service, get_kb_service
from datetime import datetime
import os
import sys
import subprocess
import requests
import shutil
import json


@api_bp.route('/api/start_service', methods=['POST'])
def start_service():
    """启动本地模型服务 (Ollama / vLLM)"""
    service = request.json.get('service')
    try:
        if service == 'ollama':
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            return jsonify({'message': 'Ollama 启动命令已发送到后台'})
        elif service == 'vllm':
            model = Config.VLLM_MODEL_NAME
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
        if ":" in model_name and not "/" in model_name:
            subprocess.Popen(["ollama", "pull", model_name], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            return jsonify({'message': f'已在后台执行: ollama pull {model_name}'})
        else:
            try:
                hf_mirror_url = f"https://hf-mirror.com/api/models/{model_name}"
                check_response = requests.get(hf_mirror_url, timeout=5)
                if check_response.status_code != 200:
                    return jsonify({'error': f'校验失败：在 hf-mirror.com 上未找到模型 [{model_name}]，请检查 ID 是否正确。状态码: {check_response.status_code}'}), 404
            except requests.exceptions.RequestException as e:
                return jsonify({'error': f'校验失败：无法连接到 hf-mirror.com，请检查网络。({str(e)})'}), 500

            safe_model_name = model_name.replace('/', '--')
            local_dir = os.path.join(Config.HF_MODEL_DIR, safe_model_name)
            os.makedirs(local_dir, exist_ok=True)

            env = os.environ.copy()
            env["HF_ENDPOINT"] = "https://hf-mirror.com"

            hf_cli_cmd = shutil.which("huggingface-cli")
            if hf_cli_cmd:
                cmd = [hf_cli_cmd, "download", model_name, "--local-dir", local_dir]
            else:
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


def _get_model_type(model_name):
    """根据模型名称后缀识别模型类型"""
    if '_awq_4bit' in model_name or '_awq_' in model_name:
        return 'quantized_awq'
    elif '_gptq' in model_name:
        return 'quantized_gptq'
    elif model_name.endswith('_merged'):
        return 'merged'
    elif model_name.endswith('_lora'):
        return 'lora'
    else:
        return 'finetuned'


@api_bp.route('/ollama/models', methods=['GET'])
def get_ollama_models():
    """获取本地 Ollama 或 vLLM 或 HuggingFace 正在运行/缓存的模型列表"""
    try:
        ollama_models_list = []
        vllm_models_list = []
        hf_models_list = []
        vllm_models_detail = []
        hf_models_detail = []

        # 尝试从 finetune_routes 导入 active_vllm_processes（懒导入）
        deployed_models = set()
        try:
            from app.api.finetune_routes import active_vllm_processes
            for service_id, info in active_vllm_processes.items():
                model_path = info.get('model_path', '')
                if model_path:
                    # 从路径中提取模型名
                    model_name = os.path.basename(model_path)
                    deployed_models.add(model_name)
        except Exception:
            pass

        try:
            response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=1)
            if response.status_code == 200:
                models = response.json().get('models', [])
                models.sort(key=lambda x: x.get('size', 0))
                ollama_models_list.extend([m['name'] for m in models])
        except requests.exceptions.RequestException:
            pass
        except Exception as e:
            current_app.logger.debug(f"Ollama 模型列表解析异常: {e}")

        # 从 vLLM API 获取已部署的模型
        vllm_api_models = set()
        try:
            vllm_base = Config.VLLM_API_URL.split('/chat/completions')[0]
            if not vllm_base.endswith('/v1'):
                vllm_base = vllm_base.rstrip('/') + '/v1'

            vllm_resp = requests.get(f"{vllm_base}/models", timeout=1)
            if vllm_resp.status_code == 200:
                vllm_models = vllm_resp.json().get('data', [])
                vllm_models_list.extend([f"vllm: {m['id']}" for m in vllm_models if 'id' in m])
                vllm_api_models.update(m['id'] for m in vllm_models if 'id' in m)
                # 添加 detail 信息
                for m in vllm_models:
                    if 'id' in m:
                        model_name = m['id']
                        vllm_models_detail.append({
                            'name': f"vllm: {model_name}",
                            'raw_name': model_name,
                            'type': 'base',  # 从API获取的通常是基座模型
                            'deployed': True
                        })
        except requests.exceptions.RequestException:
            pass
        except Exception as e:
            current_app.logger.debug(f"vLLM 模型列表解析异常: {e}")

        # 扫描本地 finetuned_models 目录
        try:
            models_dir = os.path.join(current_app.root_path, '..', 'finetuned_models')
            if os.path.exists(models_dir):
                for d in os.listdir(models_dir):
                    full_path = os.path.join(models_dir, d)
                    if os.path.isdir(full_path) and d not in ['base_models', 'runs'] and not d.endswith('_lora') and not d.endswith('_merged'):
                        vllm_tag = f"vllm: {d}"
                        if vllm_tag not in vllm_models_list:
                            vllm_models_list.append(vllm_tag)
                        
                        # 添加 detail 信息
                        model_type = _get_model_type(d)
                        is_deployed = d in deployed_models or d in vllm_api_models
                        vllm_models_detail.append({
                            'name': vllm_tag,
                            'raw_name': d,
                            'type': model_type,
                            'deployed': is_deployed
                        })
        except Exception as e:
            current_app.logger.debug(f"扫描本地离线 vLLM 目录失败: {e}")

        # 扫描本地 HuggingFace 目录
        try:
            hf_dir = os.path.join(current_app.root_path, '..', 'hugface')
            if os.path.exists(hf_dir):
                for d in os.listdir(hf_dir):
                    model_path = os.path.join(hf_dir, d)
                    if os.path.isdir(model_path):
                        if any(f.endswith('.json') or f.endswith('.safetensors') for f in os.listdir(model_path)):
                            original_id = d.replace('--', '/')
                            hf_models_list.append(original_id)
                            # 添加 detail 信息
                            hf_models_detail.append({
                                'name': original_id,
                                'type': 'hf_local',
                                'path': f"hugface/{d}"
                            })
        except Exception as e:
            current_app.logger.debug(f"扫描本地 HF 目录失败: {e}")

        online_models = [
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
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "deepseek-ai/deepseek-llm-7b-chat",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "THUDM/glm-4-9b-chat",
            "THUDM/chatglm3-6b",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-Nemo-Instruct-2407",
            "google/gemma-2-2b-it",
            "google/gemma-2-9b-it",
            "google/gemma-2-27b-it",
            "01-ai/Yi-1.5-6B-Chat",
            "01-ai/Yi-1.5-9B-Chat",
            "01-ai/Yi-1.5-34B-Chat",
            "baichuan-inc/Baichuan2-7B-Chat",
            "baichuan-inc/Baichuan2-13B-Chat"
        ]

        return jsonify({
            'models': ollama_models_list + vllm_models_list + hf_models_list,
            'ollama_models': ollama_models_list,
            'vllm_models': vllm_models_list,
            'hf_models': hf_models_list,
            'vllm_models_detail': vllm_models_detail,
            'hf_models_detail': hf_models_detail,
            'online_models': online_models,
            'query_online_models': Config.ONLINE_QUERY_MODELS
        })

    except Exception as e:
        current_app.logger.error(f'Error fetching models: {str(e)}')
        return jsonify({
            'models': [],
            'ollama_models': [],
            'vllm_models': [],
            'hf_models': [],
            'vllm_models_detail': [],
            'hf_models_detail': [],
            'online_models': [],
            'query_online_models': Config.ONLINE_QUERY_MODELS
        })


@api_bp.route('/api/search_models', methods=['GET'])
def search_models():
    """通过 HuggingFace 国内镜像搜索模型"""
    try:
        query = request.args.get('q', '').strip()
        if not query or len(query) < 2:
            return jsonify({'error': '搜索关键词至少需要2个字符'}), 400
        
        # 使用 HuggingFace API 搜索（通过国内镜像加速）
        search_url = 'https://hf-mirror.com/api/models'
        params = {
            'search': query,
            'filter': 'text-generation',  # 只搜索文本生成模型
            'sort': 'downloads',           # 按下载量排序
            'direction': '-1',             # 降序
            'limit': 30                    # 限制返回数量
        }
        
        resp = requests.get(search_url, params=params, timeout=10)
        
        if resp.status_code == 200:
            results = resp.json()
            models = []
            for item in results:
                model_id = item.get('modelId') or item.get('id', '')
                if model_id:
                    models.append({
                        'id': model_id,
                        'downloads': item.get('downloads', 0),
                        'likes': item.get('likes', 0)
                    })
            return jsonify({'models': models})
        else:
            return jsonify({'error': f'搜索服务响应异常: {resp.status_code}'}), 502
            
    except requests.exceptions.Timeout:
        return jsonify({'error': '搜索超时，请稍后重试'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'网络请求失败: {str(e)}'}), 502
    except Exception as e:
        current_app.logger.error(f'搜索模型失败: {str(e)}')
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500


@api_bp.route('/ollama/models', methods=['DELETE'])
def delete_ollama_model():
    """删除本地 Ollama 或 vLLM 或 HuggingFace 模型"""
    data = request.json
    model_name = data.get('model_name')
    model_type = data.get('model_type', 'ollama')

    if not model_name:
        return jsonify({'error': '未提供模型名称'}), 400

    try:
        if model_type == 'vllm':
            real_model_name = model_name.replace('vllm: ', '') if model_name.startswith('vllm: ') else model_name
            models_dir = os.path.join(current_app.root_path, '..', 'finetuned_models')
            target_dir = os.path.join(models_dir, real_model_name)
            
            current_app.logger.info(f'尝试删除 vLLM 模型: {real_model_name}, 路径: {target_dir}')

            if os.path.exists(target_dir) and os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
                return jsonify({'message': f'vLLM 模型 {real_model_name} 已成功删除'})
            else:
                # 尝试查找部分匹配
                if os.path.exists(models_dir):
                    available_models = os.listdir(models_dir)
                    current_app.logger.warning(f'模型目录不存在: {target_dir}, 可用模型: {available_models}')
                    return jsonify({'error': f'未找到模型目录: {real_model_name}。可用模型: {available_models}'}), 404
                else:
                    return jsonify({'error': f'finetuned_models 目录不存在: {models_dir}'}), 404
        elif model_type == 'hf':
            safe_model_name = model_name.replace('/', '--')
            target_dir = os.path.join(current_app.root_path, '..', 'hugface', safe_model_name)

            if not os.path.exists(target_dir):
                target_dir = os.path.join(current_app.root_path, '..', 'hugface', model_name)

            if os.path.exists(target_dir) and os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
                return jsonify({'message': f'Hugging Face 模型缓存 {model_name} 已成功删除'})
            else:
                return jsonify({'error': '未找到对应的本地模型目录，无法删除该 Hugging Face 模型'}), 404
        else:
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
    """设置当前使用的模型"""
    data = request.json
    model_type = data.get('model_type')
    model_name = data.get('model_name')

    if not model_type:
        return jsonify({'error': 'Missing model_type'}), 400

    try:
        if model_type == 'ollama':
            Config.ACTIVE_LLM = 'ollama'
            if model_name:
                # 去掉 ollama: 前缀（如果存在）
                Config.OLLAMA_MODEL_NAME = model_name.replace('ollama: ', '') if model_name.startswith('ollama: ') else model_name
        elif model_type == 'vllm':
            Config.ACTIVE_LLM = 'vllm'
            if model_name:
                # 去掉 vllm: 前缀（如果存在）
                Config.VLLM_MODEL_NAME = model_name.replace('vllm: ', '') if model_name.startswith('vllm: ') else model_name
        elif model_type.startswith('online_'):
            Config.ACTIVE_LLM = model_type
        else:
            return jsonify({'error': f'Unsupported model type: {model_type}'}), 400

        return jsonify({
            'message': 'Model updated',
            'current': Config.ACTIVE_LLM,
            'ollama_model': Config.OLLAMA_MODEL_NAME if model_type == 'ollama' else None,
            'vllm_model': Config.VLLM_MODEL_NAME if model_type == 'vllm' else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/llm/current', methods=['GET'])
def get_current_llm():
    """获取当前正在使用的大模型配置"""
    llm_type = Config.ACTIVE_LLM
    category = 'online'
    if llm_type == 'ollama':
        category = 'ollama'
        current_model = f"Ollama ({Config.OLLAMA_MODEL_NAME})"
    elif llm_type == 'vllm':
        category = 'vllm'
        current_model = f"vLLM ({Config.VLLM_MODEL_NAME})"
    else:
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
    user_id = request.remote_addr
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        from flask import stream_with_context
        qa_service = get_qa_service()

        def safe_stream():
            try:
                for chunk in qa_service.stream_answer_question(question, user_id=user_id, db_names=db_names, enable_tools=enable_tools):
                    yield chunk
            except Exception as inner_e:
                import traceback
                import json
                current_app.logger.error(f"Stream error: {traceback.format_exc()}")
                yield json.dumps({"type": "chunk", "content": f"\n\n> [!ERROR] 知识库检索或生成时发生错误: {str(inner_e)}\n\n"}) + "\n"

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
    correct_answer = data.get('correct_answer')
    user_id = request.remote_addr

    if not score or not question:
        return jsonify({'error': 'Missing parameters'}), 400

    if int(score) < 4:
        if correct_answer:
            try:
                qa_service = get_qa_service()
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
            qa_service = get_qa_service()
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

    import time
    time.sleep(2)

    if script_name == 'restart_service':
        return jsonify({'status': 'success', 'message': '✅ 服务重启脚本执行成功！SSH 命令 [systemctl restart app] 已下发。相关告警应在 5 分钟内清除。'})
    else:
        return jsonify({'status': 'error', 'message': f'未知的脚本名称: {script_name}'})
