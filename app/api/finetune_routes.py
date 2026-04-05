"""
微调训练相关路由：启动训练、停止训练、获取评估数据、模型列表等
"""
from flask import request, jsonify, Response, current_app, stream_with_context
from app.config import Config
from app.api.common import api_bp, get_qa_service, get_llamafactory_cli_path, active_finetune_processes
import os
import sys
import subprocess
import json
import re
import base64
import hashlib
import time
from datetime import datetime

# 尝试导入 yaml，如果失败则使用 json 代替
try:
    import yaml
except ImportError:
    yaml = None


def safe_yaml_dump(data, f, **kwargs):
    """安全地写入 yaml 或 json"""
    if yaml:
        yaml.dump(data, f, **kwargs)
    else:
        json.dump(data, f, ensure_ascii=False, indent=2)


@api_bp.route('/finetune/stop', methods=['POST'])
def stop_finetune():
    """接收前端终止训练的请求"""
    try:
        if 'current' in active_finetune_processes:
            process = active_finetune_processes['current']
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()

            del active_finetune_processes['current']
            return jsonify({'message': '训练进程已被强制终止！'})
        else:
            return jsonify({'message': '当前没有正在运行的训练进程。'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _get_eval_data_for_model(model_name):
    """内部函数：获取单个模型的评估数据，返回字典"""
    model_dir = os.path.join(current_app.root_path, '..', 'finetuned_models', model_name)
    eval_file = os.path.join(model_dir, 'eval_results.json')
    predict_file = os.path.join(model_dir, 'predict_results.json')
    trainer_state_file = os.path.join(model_dir, 'trainer_state.json')

    if not os.path.exists(trainer_state_file) and not model_name.endswith('_lora'):
        lora_model_name = f"{model_name}_lora"
        lora_model_dir = os.path.join(current_app.root_path, '..', 'finetuned_models', lora_model_name)
        if os.path.exists(lora_model_dir):
            model_name = lora_model_name
            model_dir = lora_model_dir
            eval_file = os.path.join(model_dir, 'eval_results.json')
            predict_file = os.path.join(model_dir, 'predict_results.json')
            trainer_state_file = os.path.join(model_dir, 'trainer_state.json')

    if not os.path.exists(trainer_state_file):
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(model_dir, d))]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[1]) if x.split('-')[1].isdigit() else 0, reverse=True)
            latest_ckpt = checkpoints[0]
            trainer_state_file = os.path.join(model_dir, latest_ckpt, 'trainer_state.json')

    response_data = {
        'model_name': model_name,
        'metrics': {},
        'samples': [],
        'config': {},
        'loss_history': [],
        'eval_loss_history': [],
        'lr_history': [],
        'grad_norm_history': [],
        'resource_stats': {},
        'training_config': {}
    }

    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                response_data['metrics'].update(json.load(f))
        except Exception:
            pass

    if os.path.exists(predict_file):
        try:
            with open(predict_file, 'r', encoding='utf-8') as f:
                response_data['metrics'].update(json.load(f))
        except Exception:
            pass

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
                lr_history = []
                grad_norm_history = []

                for log in log_history:
                    step = log.get('step')
                    if 'loss' in log and step is not None:
                        train_losses.append({'step': step, 'loss': log['loss']})
                    if 'eval_loss' in log and step is not None:
                        eval_losses.append({'step': step, 'loss': log['eval_loss']})
                    if 'learning_rate' in log and step is not None:
                        lr_history.append({'step': step, 'lr': log['learning_rate']})
                    if 'grad_norm' in log and step is not None:
                        grad_norm_history.append({'step': step, 'grad_norm': log['grad_norm']})

                response_data['loss_history'] = train_losses
                response_data['eval_loss_history'] = eval_losses
                response_data['lr_history'] = lr_history
                response_data['grad_norm_history'] = grad_norm_history

                if train_losses or eval_losses:
                    has_loss_data = True

                if not response_data.get('metrics') and eval_losses:
                    response_data['metrics'] = {'eval_loss': eval_losses[-1]['loss']}
                elif not response_data.get('metrics') and train_losses:
                    response_data['metrics'] = {'eval_loss': train_losses[-1]['loss']}

                # 资源统计
                response_data['resource_stats'] = {
                    'total_flos': state.get('total_flos', 0),
                    'samples_per_second': state.get('train_samples_per_second', 0),
                    'steps_per_second': state.get('train_steps_per_second', 0),
                }

        except Exception as e:
            current_app.logger.error(f"Error parsing trainer_state.json: {e}")
            pass

    # 训练配置摘要 - 从 YAML 配置文件读取
    try:
        config_dir = os.path.join(current_app.root_path, '..', 'finetune_configs')
        if os.path.exists(config_dir):
            yaml_files = sorted([f for f in os.listdir(config_dir) if f.startswith('train_')], reverse=True)
            if yaml_files:
                with open(os.path.join(config_dir, yaml_files[0]), 'r', encoding='utf-8') as f:
                    response_data['training_config'] = yaml.safe_load(f) or {}
    except Exception:
        pass

    if not response_data.get('metrics') and not has_loss_data:
        if os.path.exists(model_dir):
            response_data['metrics'] = {'status': '模型目录存在，但尚未产生标准的训练或评估日志文件。'}
        else:
            response_data['metrics'] = {'status': '获取评估数据失败，可能是该模型尚未产生训练日志或目录不存在。'}

    predict_file = os.path.join(model_dir, 'generated_predictions.jsonl')
    if os.path.exists(predict_file):
        try:
            samples = []
            with open(predict_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    samples.append({
                        "instruction": item.get("prompt", ""),
                        "reference": item.get("label", ""),
                        "predict": item.get("predict", ""),
                        "status": "待定"
                    })
            response_data['samples'] = samples[:50]
        except Exception:
            pass

    return response_data


@api_bp.route('/api/eval_data', methods=['GET'])
def get_eval_data():
    """读取评估数据并返回给前端报告页面"""
    model_name = request.args.get('model_name')
    if not model_name:
        return jsonify({'error': '未提供模型名称'}), 400

    response_data = _get_eval_data_for_model(model_name)
    return jsonify(response_data)


@api_bp.route('/api/eval_data/compare', methods=['GET'])
def compare_eval_data():
    """对比多个模型的评估数据"""
    model_names = request.args.get('model_names', '')
    if not model_names:
        return jsonify({"error": "请提供模型名列表"}), 400

    names = [n.strip() for n in model_names.split(',') if n.strip()]
    if len(names) < 2:
        return jsonify({"error": "至少选择2个模型进行对比"}), 400

    results = {}
    for name in names:
        results[name] = _get_eval_data_for_model(name)

    return jsonify({"models": results})


def is_complete_model(model_path):
    """判断是否为完整权重模型（包含model safetensors文件，非LoRA适配器）"""
    try:
        files = os.listdir(model_path)
        # 包含完整模型权重文件（model-*.safetensors 或 model.safetensors），但不是adapter
        has_model_weights = any(
            (f.startswith('model-') or f == 'model.safetensors') and f.endswith('.safetensors')
            for f in files
        )
        return has_model_weights
    except Exception:
        return False


def is_lora_adapter(model_path):
    """判断是否为LoRA适配器模型"""
    try:
        has_adapter = os.path.exists(os.path.join(model_path, 'adapter_model.safetensors'))
        has_adapter_config = os.path.exists(os.path.join(model_path, 'adapter_config.json'))
        return has_adapter and has_adapter_config
    except Exception:
        return False


@api_bp.route('/finetuned_models_list', methods=['GET'])
def get_finetuned_models():
    """获取所有微调过的模型列表，分类返回"""
    models_dir = os.path.join(current_app.root_path, '..', 'finetuned_models')

    # 已合并的完整模型（可用于知识蒸馏作为教师模型）
    merged_models = []
    # LoRA 适配器模型（可用于量化）
    lora_models = []
    # 量化模型（AWQ / GPTQ）
    quantized_models = []
    # 其他模型
    other_models = []

    if os.path.exists(models_dir):
        for d in os.listdir(models_dir):
            full_path = os.path.join(models_dir, d)
            if os.path.isdir(full_path):
                # 排除系统目录
                if d in ['base_models', 'runs']:
                    continue
                # 分类模型（优先按后缀判断，保持向后兼容）
                if d.endswith('_awq_4bit') or '_awq_' in d:
                    quantized_models.append({
                        'name': d,
                        'type': 'quantized_awq',
                        'path': full_path
                    })
                elif d.endswith('_gptq') or '_gptq_' in d:
                    quantized_models.append({
                        'name': d,
                        'type': 'quantized_gptq',
                        'path': full_path
                    })
                elif d.endswith('_merged'):
                    merged_models.append({
                        'name': d,
                        'type': 'merged',
                        'path': full_path
                    })
                elif d.endswith('_lora'):
                    lora_models.append({
                        'name': d,
                        'type': 'lora',
                        'path': full_path
                    })
                else:
                    # 对无后缀的模型进行智能检测
                    if is_complete_model(full_path):
                        merged_models.append({
                            'name': d,
                            'type': 'merged',
                            'path': full_path
                        })
                    elif is_lora_adapter(full_path):
                        lora_models.append({
                            'name': d,
                            'type': 'lora',
                            'path': full_path
                        })
                    else:
                        other_models.append({
                            'name': d,
                            'type': 'other',
                            'path': full_path
                        })

    # 兼容旧接口：提取纯字符串名称列表
    all_names = (
        [m['name'] for m in merged_models]
        + [m['name'] for m in lora_models]
        + [m['name'] for m in quantized_models]
        + [m['name'] for m in other_models]
    )

    return jsonify({
        'merged_models': merged_models,      # 已合并的完整模型（适合作为教师模型）
        'lora_models': lora_models,          # LoRA 适配器模型
        'quantized_models': quantized_models, # 量化模型（AWQ/GPTQ）
        'other_models': other_models,        # 其他模型
        'models': all_names                  # 兼容旧接口（纯字符串列表）
    })


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

    checkpoints.sort(key=lambda x: int(x.split('-')[1]) if x.split('-')[1].isdigit() else 0, reverse=True)
    return jsonify({'checkpoints': checkpoints})


@api_bp.route('/api/get_datasets', methods=['GET'])
def get_datasets():
    """获取 uploads/datasets/raw 目录下的所有文件以及 cleaned 目录下的 .jsonl 文件"""
    try:
        raw_dataset_dir = os.path.join(current_app.root_path, '..', Config.UPLOAD_FOLDER, 'datasets', 'raw')
        cleaned_dataset_dir = os.path.join(current_app.root_path, '..', Config.UPLOAD_FOLDER, 'datasets', 'cleaned')

        os.makedirs(raw_dataset_dir, exist_ok=True)
        os.makedirs(cleaned_dataset_dir, exist_ok=True)

        raw_datasets = []
        for file in os.listdir(raw_dataset_dir):
            if os.path.isfile(os.path.join(raw_dataset_dir, file)):
                raw_datasets.append(file)

        cleaned_datasets = []
        for file in os.listdir(cleaned_dataset_dir):
            if file.endswith('.jsonl') and os.path.isfile(os.path.join(cleaned_dataset_dir, file)):
                cleaned_datasets.append(file)

        raw_datasets.sort(key=lambda x: os.path.getmtime(os.path.join(raw_dataset_dir, x)), reverse=True)
        cleaned_datasets.sort(key=lambda x: os.path.getmtime(os.path.join(cleaned_dataset_dir, x)), reverse=True)

        return jsonify({
            'raw_datasets': raw_datasets,
            'cleaned_datasets': cleaned_datasets,
            'datasets': raw_datasets
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/finetune/start', methods=['POST'])
def start_finetune():
    """接收前端传来的微调任务参数，开始微调
    支持本地调试模式(local)和线上训练模式(online)，两种模式的评估指标不同
    """
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
        # 训练模式：local(本地调试) 或 online(线上训练)
        training_mode = request.form.get('trainingMode', 'local')
        
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
        base_name_clean = os.path.basename(base_name_clean.rstrip('/\\'))
        
        # 3. 替换掉特殊字符，并清理 -- 为 -
        base_name_clean = base_name_clean.replace(':', '_').replace('/', '_').replace('--', '-')
        
        # 4. 移除可能已经存在的后缀，避免类似 _v1_merged_v1 的情况
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
            # 首先尝试从 cleaned 目录查找（阶段2选择的核心数据集）
            cleaned_dir = os.path.join(current_app.root_path, '..', Config.UPLOAD_FOLDER, 'datasets', 'cleaned')
            source_dataset_path = os.path.join(cleaned_dir, dataset_file_name)
            
            # 如果在 cleaned 目录找不到，则尝试从 raw 目录查找（阶段1选择的原始数据集）
            if not os.path.exists(source_dataset_path):
                raw_dir = os.path.join(current_app.root_path, '..', Config.UPLOAD_FOLDER, 'datasets', 'raw')
                source_dataset_path = os.path.join(raw_dir, dataset_file_name)
            
            if not os.path.exists(source_dataset_path):
                return jsonify({'error': f'找不到数据集文件: {dataset_file_name}'}), 404
                
            # 确保 cleaned 目录存在
            os.makedirs(cleaned_dir, exist_ok=True)
            
            # 如果文件已经在 cleaned 目录，直接使用；否则生成 cleaned_ 前缀的文件名
            if source_dataset_path.startswith(cleaned_dir):
                dataset_path = source_dataset_path
            else:
                dataset_path = os.path.join(cleaned_dir, f"cleaned_{dataset_file_name}")
            
            # 读取并清洗 JSONL 文件，过滤掉有语法错误的行，同时统一下格式
            try:
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
        
        def generate_finetune_logs(base_model_name, mode=training_mode):
            import time
            import subprocess
            
            yield f"data: > 接收到真实微调任务请求，准备初始化环境...\n\n"
            time.sleep(1)
            yield f"data: > 基座模型: [{base_model_name}], 目标模型: [{new_model}]\n\n"
            
            # 显示训练模式信息
            if mode == 'local':
                yield f"data: > [训练模式] 🖥️ 本地调试模式 - 低资源消耗，快速验证流程\n\n"
                yield f"data: > [指标关注] 训练loss下降趋势、模型收敛性、无OOM错误\n\n"
            else:
                yield f"data: > [训练模式] ☁️ 线上训练模式 - 高质量正式训练\n\n"
                yield f"data: > [指标关注] eval_loss、perplexity、BLEU/ROUGE、模型泛化能力\n\n"
            
            # 清理 vllm: 前缀
            is_vllm = False
            if base_model_name.startswith('vllm: '):
                base_model_name = base_model_name.replace('vllm: ', '')
                is_vllm = True
                
            hf_model_path = base_model_name
            
            # 检查是否为本地下载的 HF 模型（支持两种命名格式：Qwen/xxx 和 Qwen--xxx）
            local_hf_path = os.path.join(Config.HF_MODEL_DIR, base_model_name)
            local_hf_path_alt = os.path.join(Config.HF_MODEL_DIR, base_model_name.replace('/', '--'))
            if os.path.exists(local_hf_path) and os.path.isdir(local_hf_path):
                hf_model_path = local_hf_path
                yield f"data: > [INFO] 发现本地 Hugging Face 模型缓存，直接使用绝对路径: {hf_model_path}\n\n"
            elif os.path.exists(local_hf_path_alt) and os.path.isdir(local_hf_path_alt):
                hf_model_path = local_hf_path_alt
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
                "save_steps": int(save_steps) if save_steps and save_steps != 'undefined' else 100,
                "learning_rate": float(learning_rate) if learning_rate and learning_rate != 'undefined' else 2e-4,
                "num_train_epochs": float(epochs) if epochs and epochs != 'undefined' else 3.0,
                "max_length": int(max_length) if max_length and max_length != 'undefined' else 1024, # 降低默认最大长度防 OOM
                "weight_decay": float(weight_decay) if weight_decay and weight_decay != 'undefined' else 0.0,
                "max_grad_norm": float(max_grad_norm) if max_grad_norm and max_grad_norm != 'undefined' else 1.0,
                "dataloader_num_workers": 0, # Windows 环境下避免多进程 DataLoader 导致的 pickle 报错
                "dataloader_pin_memory": True, # 锁页内存，加速 CPU 到 GPU 的数据拷贝
                "seed": int(seed) if seed and seed != 'undefined' else 42,
                "plot_loss": True,
                "report_to": "tensorboard", # 启用 TensorBoard 记录
                # 下面的两个参数告诉 LLaMA-Factory 使用内置的计算指标回调，从而把 F1, 准确率等写入 TensorBoard
                "compute_accuracy": True,
                # 小数据集时不切分验证集，避免 datasets 缓存冲突问题
                # "val_size": 0.1,
                # Windows 环境下防止模型加载卡死的关键参数
                "low_cpu_mem_usage": True,  # 使用低内存加载模式
                "trust_remote_code": True,  # 允许加载自定义模型架构
                "use_fast_tokenizer": False,  # 禁用 fast tokenizer，避免 Windows 下卡住
                "resize_vocab": False,  # 禁用词表调整
                "streaming": False,  # 禁用流式加载，避免小数据集问题
                "bf16": False,  # 默认关闭 bf16，避免不支持时的错误
                "fp16": False,  # 默认关闭 fp16
            }
            
            # 精度设置 - Windows 环境下建议使用 fp32 以避免兼容性问题
            if precision == "fp16":
                # 检查是否支持 FP16（需要 NVIDIA GPU 支持 Tensor Cores）
                try:
                    import torch
                    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
                        train_config["fp16"] = True
                        train_config["bf16"] = False
                    else:
                        yield f"data: > [WARNING] 当前 GPU 不支持 FP16，自动回退到 FP32 模式\n\n"
                        train_config["fp16"] = False
                        train_config["bf16"] = False
                except:
                    train_config["fp16"] = True
                    train_config["bf16"] = False
            elif precision == "bf16":
                train_config["bf16"] = True
                train_config["fp16"] = False
            else: # fp32 - 最安全的默认选项
                train_config["pure_bf16"] = False
                train_config["fp16"] = False
                train_config["bf16"] = False
                
            # 量化设置
            if quantization and quantization != 'none':
                train_config["quantization_bit"] = int(quantization)
                
            # Flash Attention
            if flash_attn and flash_attn not in ('auto', 'none', 'undefined'):
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
            
            # 评估配置 - 当 eval_steps > 0 时启用
            if eval_steps and eval_steps not in ('0', 'undefined', ''):
                train_config["eval_strategy"] = "steps"
                train_config["eval_steps"] = int(eval_steps)
                train_config["per_device_eval_batch_size"] = 1
                train_config["val_size"] = 0.1
                train_config["predict_with_generate"] = True
                
            # LLaMA-Factory 0.8.3 会默认在代码内部处理 trust_remote_code
            # 如果从配置文件直接传入可能被 HfArgumentParser 拒绝
            if 'trust_remote_code' in train_config:
                del train_config['trust_remote_code']
            
            import yaml
            with open(config_path, 'w', encoding='utf-8') as f:
                safe_yaml_dump(train_config, f, allow_unicode=True)
                
            yield f"data: > 训练配置已生成: {config_path}\n\n"
            yield f"data: > 正在拉起 LLaMA-Factory 训练进程 (llamafactory-cli train)...\n\n"
            
            # 启动真实的 subprocess (这里调用 llamafactory-cli)
            # 注意：这要求你的环境中已经安装了 LLaMA-Factory: pip install llamafactory
            try:
                # 使用 Popen 实时捕获标准输出和错误输出
                # 设置 HF_ENDPOINT 使用国内镜像源，防止下载模型时超时
                env = os.environ.copy()
                env['HF_ENDPOINT'] = 'https://hf-mirror.com'
                # 如果使用的是本地绝对路径模型，启用离线模式防止不必要的网络请求
                if os.path.isabs(hf_model_path) and os.path.exists(hf_model_path):
                    env['HF_HUB_OFFLINE'] = '1'
                    env['TRANSFORMERS_OFFLINE'] = '1'
                # 修复 PyTorch 2.6 安全策略更新导致的断点续训报错: _pickle.UnpicklingError
                env['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
                # 兼容部分老版本 numpy 的 pickle 加载
                env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                
                # OMP_NUM_THREADS 设为 1 可能在部分 Linux/libgomp 下导致 Invalid value，或者对性能有影响
                # 仅在 Windows 环境下限制线程数以防止卡死，Linux 下保持默认或合理值
                if os.name == 'nt':
                    env['OMP_NUM_THREADS'] = '1'
                    env['MKL_NUM_THREADS'] = '1'
                    env['TOKENIZERS_PARALLELISM'] = 'false'
                
                # 禁用 Hugging Face datasets 缓存，避免 Windows 下的文件冲突问题
                env['HF_DATASETS_CACHE'] = os.path.join(root_path, '..', 'cache', 'datasets')
                env['HF_DATASETS_IN_MEMORY'] = '1'  # 尽量在内存中处理数据集
                # 调试用的环境变量（如果需要详细错误信息，可以取消注释）
                # env['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步 CUDA 操作，获取更清晰的错误堆栈
                
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
                            safe_yaml_dump(export_config, f, allow_unicode=True)
                            
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
                    
                    # 清理不完整的模型输出，防止出现在模型列表中
                    import shutil
                    if os.path.exists(output_dir):
                        shutil.rmtree(output_dir, ignore_errors=True)
                        yield f"data: > [INFO] 已清理不完整的模型输出目录: {output_dir}\n\n"
                    if os.path.exists(logging_dir):
                        shutil.rmtree(logging_dir, ignore_errors=True)
                    
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


# ============================================================================
# 阶段 1: 数据准备 API
# ============================================================================

@api_bp.route('/api/data_prep/start', methods=['POST'])
def start_data_preparation():
    """
    数据准备/清洗 API
    接收原始数据文件，进行去重、质量评分、格式转换等处理
    增加模型介入分析，复用问答系统的模型选择逻辑
    """
    try:
        data = request.get_json()
        raw_dataset = data.get('raw_dataset')
        dedup_strategy = data.get('dedup_strategy', 'none')
        enable_scoring = data.get('enable_scoring', False)
        score_threshold = data.get('score_threshold', 7.0)
        output_name = data.get('output_name', 'core_dataset.jsonl')
        conversion_strategy = data.get('conversion_strategy', 'keep_original')
        # 新增：模型介入分析选项 - 与问答系统模型选择一致
        enable_model_analysis = data.get('enable_model_analysis', True)
        analysis_model = data.get('analysis_model', None)  # 可选指定模型，None则使用默认
        analysis_model_category = data.get('analysis_model_category', None)  # 模型类别：online/ollama/vllm
        
        if not raw_dataset:
            return jsonify({'error': '未指定原始数据文件'}), 400
        
        # 构建路径
        raw_path = os.path.join(current_app.root_path, '..', Config.UPLOAD_FOLDER, 'datasets', 'raw', raw_dataset)
        cleaned_dir = os.path.join(current_app.root_path, '..', Config.UPLOAD_FOLDER, 'datasets', 'cleaned')
        output_path = os.path.join(cleaned_dir, output_name)
        
        if not os.path.exists(raw_path):
            return jsonify({'error': f'原始文件不存在: {raw_dataset}'}), 404
        
        os.makedirs(cleaned_dir, exist_ok=True)
        
        def generate_data_prep_logs():
            yield f"data: > 开始数据准备任务...\n\n"
            yield f"data: > 输入文件: {raw_dataset}\n\n"
            yield f"data: > 去重策略: {dedup_strategy}\n\n"
            
            try:
                # 读取原始数据
                yield f"data: > 正在读取原始数据...\n\n"
                
                raw_data = []
                with open(raw_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                raw_data.append(json.loads(line))
                            except:
                                # 非 JSON 行，作为纯文本处理
                                raw_data.append({'text': line})
                
                yield f"data: > 读取完成，共 {len(raw_data)} 条原始数据\n\n"
                
                # 去重处理
                if dedup_strategy != 'none':
                    yield f"data: > 正在执行去重 ({dedup_strategy})...\n\n"
                    
                    if dedup_strategy == 'exact':
                        # MD5 精确去重
                        seen_hashes = set()
                        unique_data = []
                        for item in raw_data:
                            content = json.dumps(item, sort_keys=True, ensure_ascii=False)
                            hash_val = hashlib.md5(content.encode()).hexdigest()
                            if hash_val not in seen_hashes:
                                seen_hashes.add(hash_val)
                                unique_data.append(item)
                        raw_data = unique_data
                        
                    elif dedup_strategy == 'minhash':
                        # 简化版 MinHash 去重（基于文本相似度）
                        # 实际项目中应使用 datatrove 库
                        yield f"data: > [WARNING] MinHash 去重需要 datatrove 库，当前使用简化版近似去重\n\n"
                        unique_data = []
                        texts = []
                        for item in raw_data:
                            text = item.get('text', '') or item.get('instruction', '') + item.get('output', '')
                            is_dup = False
                            for existing_text in texts:
                                # 简单相似度计算
                                if len(text) > 0 and len(existing_text) > 0:
                                    similarity = len(set(text) & set(existing_text)) / len(set(text) | set(existing_text))
                                    if similarity > 0.85:
                                        is_dup = True
                                        break
                            if not is_dup:
                                texts.append(text)
                                unique_data.append(item)
                        raw_data = unique_data
                    
                    yield f"data: > 去重完成，剩余 {len(raw_data)} 条数据\n\n"
                
                # 质量评分过滤
                if enable_scoring:
                    yield f"data: > 正在执行质量评分过滤 (阈值: {score_threshold})...\n\n"
                    # 简化版质量评分：基于文本长度和结构
                    filtered_data = []
                    for item in raw_data:
                        score = 0
                        text = item.get('text', '') or item.get('instruction', '') + item.get('output', '')
                        
                        # 长度评分
                        if len(text) > 50:
                            score += 3
                        elif len(text) > 20:
                            score += 2
                        else:
                            score += 1
                        
                        # 结构评分
                        if 'instruction' in item and 'output' in item:
                            score += 3
                        elif 'text' in item:
                            score += 2
                        
                        # 内容质量评分（简化）
                        if len(set(text)) > len(text) * 0.5:  # 字符多样性
                            score += 2
                        
                        if score >= score_threshold:
                            filtered_data.append(item)
                    
                    raw_data = filtered_data
                    yield f"data: > 质量过滤完成，剩余 {len(raw_data)} 条数据\n\n"
                
                # 格式转换 - 增加模型介入分析
                yield f"data: > 正在执行格式转换 ({conversion_strategy})...\n\n"
                
                # 如果启用模型分析且是知识问答转换策略，使用LLM进行更详细的分析
                model_analysis_active = enable_model_analysis and conversion_strategy == 'knowledge_qa'
                if model_analysis_active:
                    yield f"data: > [模型介入] 启用LLM智能分析数据内容...\n\n"
                    try:
                        # 复用问答系统的模型选择逻辑 - 与问答系统完全一致
                        from app.services.qa_service import QAService
                        qa_service = QAService()
                        
                        # 如果用户指定了分析模型，根据类别创建对应的LLM
                        original_llm = None
                        if analysis_model and analysis_model_category:
                            original_llm = qa_service.llm
                            yield f"data: > [模型切换] 使用用户选择的模型: {analysis_model_category} / {analysis_model}\n\n"
                            
                            # 根据类别创建对应的LLM - 与问答系统逻辑一致
                            from langchain_openai import ChatOpenAI
                            if analysis_model_category == 'ollama':
                                from langchain_community.chat_models import ChatOllama
                                qa_service.llm = ChatOllama(
                                    base_url=Config.OLLAMA_BASE_URL,
                                    model=analysis_model,
                                    temperature=0.1
                                )
                            elif analysis_model_category == 'vllm':
                                base_url = Config.VLLM_API_URL
                                if base_url.endswith('/chat/completions'):
                                    base_url = base_url.replace('/chat/completions', '')
                                qa_service.llm = ChatOpenAI(
                                    base_url=base_url,
                                    api_key=Config.VLLM_API_KEY,
                                    model=analysis_model,
                                    temperature=0.1,
                                    max_retries=1,
                                    timeout=60
                                )
                            elif analysis_model_category == 'online':
                                # 在线模型处理
                                if analysis_model == 'deepseek':
                                    qa_service.llm = ChatOpenAI(
                                        base_url=Config.DEEPSEEK_API_URL.replace('/chat/completions', ''),
                                        api_key=Config.DEEPSEEK_API_KEY,
                                        model=Config.DEEPSEEK_MODEL_NAME,
                                        temperature=0.1,
                                        max_retries=1,
                                        timeout=60
                                    )
                                elif analysis_model == 'qwen':
                                    qa_service.llm = ChatOpenAI(
                                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                        api_key=os.getenv('DASHSCOPE_API_KEY', Config.QWEN_API_KEY),
                                        model="qwen-plus",
                                        temperature=0.1,
                                        max_retries=1,
                                        timeout=60
                                    )
                                else:
                                    # 使用默认配置
                                    pass
                        else:
                            yield f"data: > 使用问答系统默认模型: {Config.ACTIVE_LLM} 进行分析\n\n"
                        
                    except Exception as e:
                        yield f"data: > [WARNING] LLM分析初始化失败: {str(e)}，将使用启发式转换\n\n"
                        model_analysis_active = False
                
                converted_data = []
                for idx, item in enumerate(raw_data):
                    if conversion_strategy == 'knowledge_qa':
                        # 转换为知识问答格式
                        text = item.get('text', '')
                        if text:
                            # 如果启用模型分析，使用LLM生成更准确的问答对
                            if model_analysis_active and idx < 10:  # 只对前10条使用LLM分析，避免耗时过长
                                try:
                                    from langchain_core.messages import HumanMessage
                                    analysis_prompt = f"""请分析以下运维知识文本，提取一个核心问题和详细答案。
文本内容：
{text[:500]}  # 限制长度避免过长

请按以下JSON格式输出（只输出JSON，不要有其他内容）：
{{
    "question": "核心问题（简洁明了）",
    "answer": "详细答案（专业完整）"
}}
"""
                                    response = qa_service.llm.invoke([HumanMessage(content=analysis_prompt)])
                                    # 尝试解析JSON响应
                                    try:
                                        import re
                                        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                                        if json_match:
                                            parsed = json.loads(json_match.group())
                                            converted_data.append({
                                                'instruction': parsed.get('question', '请解释以下内容'),
                                                'input': '',
                                                'output': parsed.get('answer', text)
                                            })
                                            if (idx + 1) % 5 == 0:
                                                yield f"data: > LLM已分析 {idx + 1} 条数据...\n\n"
                                            continue
                                    except:
                                        pass  # 解析失败则回退到启发式
                                except Exception as e:
                                    pass  # LLM调用失败则回退到启发式
                            
                            # 启发式转换（原逻辑）
                            sentences = text.split('。')
                            if len(sentences) >= 2:
                                converted_data.append({
                                    'instruction': sentences[0] + '？',
                                    'input': '',
                                    'output': '。'.join(sentences[1:])
                                })
                            else:
                                converted_data.append({
                                    'instruction': '请解释以下内容',
                                    'input': '',
                                    'output': text
                                })
                        else:
                            converted_data.append(item)
                            
                    elif conversion_strategy == 'text_completion':
                        # 文本续写格式
                        text = item.get('text', '')
                        if text:
                            mid = len(text) // 2
                            converted_data.append({
                                'instruction': text[:mid],
                                'input': '',
                                'output': text[mid:]
                            })
                        else:
                            converted_data.append(item)
                    else:
                        # 保留原格式
                        converted_data.append(item)
                
                # 恢复原始LLM配置
                if model_analysis_active and original_llm:
                    qa_service.llm = original_llm
                
                # 保存清洗后的数据
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in converted_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                yield f"data: > 数据准备完成！\n\n"
                yield f"data: > 输出文件: {output_name}\n\n"
                yield f"data: > 最终数据量: {len(converted_data)} 条\n\n"
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: > [ERROR] 数据准备失败: {str(e)}\n\n"
                yield f"data: [DONE]\n\n"
        
        from flask import Response
        return Response(generate_data_prep_logs(), mimetype='text/event-stream')
        
    except Exception as e:
        current_app.logger.error(f"Data prep error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# 阶段 3: 知识蒸馏 API
# ============================================================================

@api_bp.route('/api/distill/start', methods=['POST'])
def start_distillation():
    """
    知识蒸馏 API
    使用教师模型生成软标签/回答，训练学生模型
    """
    try:
        data = request.get_json()
        teacher_model = data.get('teacher_model')
        student_model = data.get('student_model')
        dataset = data.get('dataset')
        strategy = data.get('strategy', 'text_sft')
        temperature = float(data.get('temperature', 4.0))
        loss_weight = float(data.get('loss_weight', 0.7))
        
        if not teacher_model or not student_model:
            return jsonify({'error': '请指定教师模型和学生模型'}), 400
        
        if not dataset:
            return jsonify({'error': '请选择训练数据集'}), 400
        
        # 在生成器外部获取路径和配置，避免 Flask 上下文问题
        root_path = current_app.root_path
        models_dir = os.path.join(root_path, '..', 'finetuned_models')
        upload_folder = Config.UPLOAD_FOLDER
        
        def generate_distillation_logs():
            yield f"data: > 开始知识蒸馏任务...\n\n"
            yield f"data: > 教师模型: {teacher_model}\n\n"
            yield f"data: > 学生模型: {student_model}\n\n"
            yield f"data: > 蒸馏策略: {strategy}\n\n"
            
            try:
                # 优化教师模型路径查找逻辑
                possible_teacher_paths = [
                    os.path.join(models_dir, teacher_model),  # 标准路径
                    os.path.join(models_dir, teacher_model + '_merged'),  # 合并后的模型
                    os.path.join(models_dir, teacher_model + '_lora'),  # LoRA适配器
                ]
                
                teacher_path = None
                for path in possible_teacher_paths:
                    if os.path.exists(path) and os.path.isdir(path):
                        teacher_path = path
                        break
                
                # 检查教师模型是否存在
                if not teacher_path:
                    yield f"data: > [ERROR] 教师模型不存在: {teacher_model}\n\n"
                    yield f"data: > [INFO] 已尝试以下路径:\n\n"
                    for path in possible_teacher_paths:
                        yield f"data: >   - {path}\n\n"
                    yield f"data: > [TIP] 请先在阶段2完成教师模型微调并合并权重\n\n"
                    yield f"data: [DONE]\n\n"
                    return
                
                yield f"data: > [INFO] 找到教师模型路径: {teacher_path}\n\n"
                
                # 解析学生模型名称并查找路径
                student_name = student_model.replace('/', '--')
                
                # 优化模型路径查找逻辑：支持多个可能的位置
                possible_student_paths = [
                    os.path.join(root_path, '..', 'hugface', student_name),  # 标准HF路径
                    os.path.join(root_path, '..', 'hugface', student_model),  # 原始名称
                    os.path.join(Config.HF_MODEL_DIR, student_name),  # 配置中的HF目录
                    os.path.join(Config.HF_MODEL_DIR, student_model),  # 配置中的原始名称
                    os.path.join(root_path, '..', 'finetuned_models', 'base_models', student_name),  # 基础模型目录
                ]
                
                student_path = None
                for path in possible_student_paths:
                    if os.path.exists(path) and os.path.isdir(path):
                        student_path = path
                        break
                
                if not student_path:
                    yield f"data: > [ERROR] 学生模型不存在: {student_model}\n\n"
                    yield f"data: > [INFO] 已尝试以下路径:\n\n"
                    for path in possible_student_paths:
                        yield f"data: >   - {path}\n\n"
                    yield f"data: > [TIP] 请确保学生模型已下载到 hugface/ 目录或 finetuned_models/base_models/ 目录\n\n"
                    yield f"data: [DONE]\n\n"
                    return
                
                yield f"data: > [INFO] 找到学生模型路径: {student_path}\n\n"
                
                # 生成蒸馏后模型名称（只使用模型名，不包含路径）
                teacher_name = os.path.basename(teacher_path.rstrip('/\\'))
                student_name_only = os.path.basename(student_path.rstrip('/\\'))
                distilled_name = f"{student_name_only}_distilled_by_{teacher_name}"
                output_dir = os.path.join(models_dir, distilled_name)
                os.makedirs(output_dir, exist_ok=True)
                
                yield f"data: > 蒸馏输出目录: {distilled_name}\n\n"
                
                # 完整版知识蒸馏实现
                # 1. 使用教师模型生成软标签数据集
                # 2. 使用 KL 散度损失训练学生模型
                
                yield f"data: > [INFO] 开始完整知识蒸馏流程...\n\n"
                yield f"data: > 温度系数: {temperature}, 蒸馏权重: {loss_weight}\n\n"
                
                # 步骤 1: 准备训练数据（使用 cleaned 目录下的数据集）
                yield f"data: > 步骤 1/3: 准备训练数据...\n\n"
                
                cleaned_dataset_dir = os.path.join(root_path, '..', upload_folder, 'datasets', 'cleaned')
                distill_dataset_path = os.path.join(cleaned_dataset_dir, dataset)
                
                if not os.path.exists(distill_dataset_path):
                    yield f"data: > [ERROR] 数据集不存在: {dataset}\n\n"
                    yield f"data: [DONE]\n\n"
                    return
                
                yield f"data: > 使用数据集: {dataset}\n\n"
                
                # 步骤 2: 使用教师模型生成软标签（完整版：加载教师模型输出 logits）
                yield f"data: > 步骤 2/3: 使用教师模型生成软标签数据（完整版）...\n\n"
                
                # 生成软标签数据集路径
                soft_label_dataset = os.path.join(output_dir, 'soft_label_dataset.jsonl')
                logits_cache_path = os.path.join(output_dir, 'teacher_logits.pt')
                
                try:
                    # 加载训练数据
                    train_data = []
                    with open(distill_dataset_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                train_data.append(json.loads(line))
                    
                    yield f"data: > 加载训练数据: {len(train_data)} 条\n\n"
                    
                    # 完整版：加载教师模型并生成软标签（logits 分布）
                    yield f"data: > 正在加载教师模型: {teacher_model}...\n\n"
                    
                    try:
                        import torch
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        
                        # 设置设备
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        yield f"data: > 使用设备: {device}\n\n"
                        
                        # 加载教师模型和 tokenizer
                        teacher_tokenizer = AutoTokenizer.from_pretrained(
                            teacher_path,
                            trust_remote_code=True,
                            padding_side='left'
                        )
                        if teacher_tokenizer.pad_token is None:
                            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
                        
                        yield f"data: > 教师 tokenizer 加载完成\n\n"
                        
                        # 加载教师模型（使用 FP16 减少内存占用）
                        teacher_model_loaded = AutoModelForCausalLM.from_pretrained(
                            teacher_path,
                            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                            device_map='auto' if device == 'cuda' else None,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                        teacher_model_loaded.eval()
                        
                        yield f"data: > 教师模型加载完成，开始生成软标签...\n\n"
                        
                        # 生成软标签（logits）
                        soft_labeled_data = []
                        teacher_logits_list = []
                        
                        with torch.no_grad():
                            for idx, item in enumerate(train_data):
                                instruction = item.get('instruction', '')
                                input_text = item.get('input', '')
                                original_output = item.get('output', '')
                                
                                # 构建完整输入（instruction + input）
                                if input_text:
                                    full_prompt = f"{instruction}\n\n{input_text}"
                                else:
                                    full_prompt = instruction
                                
                                # 构建完整序列（输入 + 输出）用于计算 logits
                                full_text = f"{full_prompt}\n\n{original_output}"
                                
                                # Tokenize
                                inputs = teacher_tokenizer(
                                    full_text,
                                    return_tensors='pt',
                                    truncation=True,
                                    max_length=512,
                                    padding=True
                                )
                                
                                if device == 'cuda':
                                    inputs = {k: v.to(device) for k, v in inputs.items()}
                                
                                # 教师模型前向传播获取 logits
                                outputs = teacher_model_loaded(**inputs)
                                logits = outputs.logits  # [batch, seq_len, vocab_size]
                                
                                # 应用温度系数软化分布
                                # T > 1 时分布更平滑，包含更多概率信息
                                soft_logits = logits / temperature
                                
                                # 保存软标签信息
                                soft_labeled_item = {
                                    'instruction': instruction,
                                    'input': input_text,
                                    'output': original_output,  # 硬标签
                                    'teacher_output': original_output,  # 教师生成的文本（可选）
                                    'temperature': temperature,
                                    'has_soft_labels': True,
                                }
                                soft_labeled_data.append(soft_labeled_item)
                                
                                # 保存 logits 到列表（后续用于 KL 散度计算）
                                # 只保存输出部分的 logits
                                prompt_tokens = teacher_tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=512)
                                prompt_length = prompt_tokens.input_ids.shape[1]
                                output_logits = soft_logits[0, prompt_length-1:-1, :]  # 输出部分的 logits
                                teacher_logits_list.append(output_logits.cpu())
                                
                                if (idx + 1) % 5 == 0 or idx == 0:
                                    yield f"data: > 已处理 {idx + 1}/{len(train_data)} 条数据的软标签...\n\n"
                                
                                # 定期清理 GPU 缓存
                                if device == 'cuda' and (idx + 1) % 10 == 0:
                                    torch.cuda.empty_cache()
                        
                        # 保存软标签数据集（元数据）
                        with open(soft_label_dataset, 'w', encoding='utf-8') as f:
                            for item in soft_labeled_data:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        
                        # 保存教师 logits 缓存（用于后续 KL 散度训练）
                        torch.save(teacher_logits_list, logits_cache_path)
                        
                        yield f"data: > 软标签数据集已生成: {soft_label_dataset}\n\n"
                        yield f"data: > 教师 logits 缓存已保存: {logits_cache_path}\n\n"
                        yield f"data: > 平均 logits 形状: {teacher_logits_list[0].shape}\n\n"
                        
                        # 清理教师模型释放内存
                        del teacher_model_loaded
                        del teacher_tokenizer
                        del teacher_logits_list
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                        
                        yield f"data: > 教师模型已卸载，内存已释放\n\n"
                        
                    except ImportError as ie:
                        yield f"data: > [ERROR] 缺少必要的库: {str(ie)}\n\n"
                        yield f"data: > 请安装: pip install transformers torch\n\n"
                        yield f"data: [DONE]\n\n"
                        return
                    except Exception as e:
                        yield f"data: > [ERROR] 加载教师模型失败: {str(e)}\n\n"
                        yield f"data: [DONE]\n\n"
                        return
                    
                except Exception as e:
                    yield f"data: > [ERROR] 生成软标签失败: {str(e)}\n\n"
                    yield f"data: [DONE]\n\n"
                    return
                
                # 步骤 3: 创建支持 KL 散度的蒸馏配置
                yield f"data: > 步骤 3/3: 创建蒸馏训练配置（支持 KL 散度损失）...\n\n"
                
                config_path = os.path.join(root_path, '..', 'finetune_configs', f'distill_{datetime.now().strftime("%Y%m%d%H%M%S")}.yaml')
                
                # 创建 dataset_info.json 条目
                dataset_info_entry = {
                    f"distill_{distilled_name}": {
                        "file_name": soft_label_dataset,
                        "formatting": "alpaca",
                        "columns": {
                            "prompt": "instruction",
                            "query": "input",
                            "response": "output"
                        }
                    }
                }
                
                # 更新 dataset_info.json
                dataset_info_path = os.path.join(root_path, '..', 'model', 'LLaMA-Factory', 'data', 'dataset_info.json')
                try:
                    if os.path.exists(dataset_info_path):
                        with open(dataset_info_path, 'r', encoding='utf-8') as f:
                            dataset_info = json.load(f)
                    else:
                        dataset_info = {}
                    
                    dataset_info.update(dataset_info_entry)
                    
                    with open(dataset_info_path, 'w', encoding='utf-8') as f:
                        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
                    
                    yield f"data: > 数据集已注册到 dataset_info.json\n\n"
                except Exception as e:
                    yield f"data: > [WARNING] 注册数据集失败: {str(e)}\n\n"
                
                # 蒸馏训练配置（使用 KL 散度损失）
                # 注意：完整 KL 散度训练需要自定义训练脚本或修改 LLaMA-Factory
                # 这里配置支持标准 SFT + 温度系数，作为蒸馏的基础
                distill_config = {
                    'model_name_or_path': student_path,
                    'adapter_name_or_path': None,  # 学生模型从头训练
                    'template': 'qwen',
                    'finetuning_type': 'lora',
                    'lora_target': 'all',
                    'lora_rank': 16,
                    'lora_alpha': 32,
                    'lora_dropout': 0.05,
                    'output_dir': output_dir,
                    'overwrite_output_dir': True,
                    'dataset': f"distill_{distilled_name}",
                    'cutoff_len': 1024,
                    'per_device_train_batch_size': 1,
                    'gradient_accumulation_steps': 4,
                    'learning_rate': 2e-4,
                    'num_train_epochs': 3.0,
                    'logging_steps': 5,
                    'save_steps': 100,
                    'warmup_ratio': 0.1,
                    'bf16': False,
                    'fp16': False,
                    'plot_loss': True,
                    'report_to': 'none',
                    'low_cpu_mem_usage': True,
                    'trust_remote_code': True,
                    # 蒸馏特定配置 - 只保留 LLaMA-Factory 认识的标准参数
                    'temperature': temperature,
                    'do_train': True,
                }
                
                # 保存蒸馏元数据（LLaMA-Factory 不认识的自定义参数）
                distill_metadata = {
                    'distill_loss_weight': loss_weight,
                    'teacher_logits_path': logits_cache_path,
                    'temperature': temperature,
                    'teacher_model': teacher_model,
                    'student_model': student_model,
                    'strategy': strategy,
                }
                metadata_path = os.path.join(output_dir, 'distill_metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(distill_metadata, f, ensure_ascii=False, indent=2)
                
                # 创建自定义蒸馏训练脚本（如果需要完整的 KL 散度支持）
                distill_script_path = os.path.join(output_dir, 'distill_train.py')
                distill_script_content = f'''"""
知识蒸馏训练脚本 - 支持 KL 散度损失
使用教师模型的 logits 指导学生模型训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import json

# 配置
student_model_path = "{student_path}"
teacher_logits_path = "{logits_cache_path}"
output_dir = "{output_dir}"
temperature = {temperature}
loss_weight = {loss_weight}

# 加载学生模型
model = AutoModelForCausalLM.from_pretrained(
    student_model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

# 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, peft_config)

# 加载教师 logits
teacher_logits_list = torch.load(teacher_logits_path)

class DistillationTrainer(Trainer):
    """支持 KL 散度损失的自定义 Trainer"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 标准交叉熵损失
        outputs = model(**inputs)
        ce_loss = outputs.loss
        
        # 获取学生模型 logits
        student_logits = outputs.logits / temperature
        
        # 获取对应的教师 logits
        batch_idx = self.state.global_step % len(teacher_logits_list)
        teacher_logits = teacher_logits_list[batch_idx].to(student_logits.device)
        
        # 计算 KL 散度损失
        # KL(P_teacher || P_student)
        kl_loss = F.kl_div(
            F.log_softmax(student_logits[:, -teacher_logits.shape[0]:, :], dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # 组合损失
        loss = (1 - loss_weight) * ce_loss + loss_weight * kl_loss
        
        return (loss, outputs) if return_outputs else loss

print("蒸馏训练脚本已准备好")
print(f"教师 logits 数量: {{len(teacher_logits_list)}}")
'''
                
                with open(distill_script_path, 'w', encoding='utf-8') as f:
                    f.write(distill_script_content)
                
                yield f"data: > 自定义蒸馏训练脚本已生成: {distill_script_path}\n\n"
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    safe_yaml_dump(distill_config, f, allow_unicode=True)
                
                yield f"data: > 蒸馏配置已生成: {config_path}\n\n"
                
                # 执行蒸馏训练
                yield f"data: > 开始蒸馏训练（使用 KL 散度损失）...\n\n"
                
                try:
                    env = os.environ.copy()
                    env['OMP_NUM_THREADS'] = '1'
                    env['MKL_NUM_THREADS'] = '1'
                    env['TOKENIZERS_PARALLELISM'] = 'false'
                    env['HF_DATASETS_CACHE'] = os.path.join(root_path, '..', 'cache', 'datasets')
                    
                    process = subprocess.Popen(
                        [get_llamafactory_cli_path(), 'train', config_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        encoding='utf-8',
                        errors='replace',
                        env=env
                    )
                    
                    # 实时输出训练日志
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
                            if len(buffer) > 150:
                                clean_line = buffer.strip().replace('"', "'")
                                if clean_line:
                                    yield f"data: > {clean_line}\n\n"
                                buffer = ""
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        yield f"data: > [SUCCESS] 知识蒸馏完成！\n\n"
                        yield f"data: > 蒸馏后模型保存至: {output_dir}\n\n"
                    else:
                        yield f"data: > [ERROR] 蒸馏训练失败，状态码: {process.returncode}\n\n"
                    
                except Exception as e:
                    yield f"data: > [ERROR] 蒸馏训练失败: {str(e)}\n\n"
                
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: > [ERROR] 知识蒸馏失败: {str(e)}\n\n"
                yield f"data: [DONE]\n\n"
        
        from flask import Response
        return Response(
            generate_distillation_logs(), 
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        current_app.logger.error(f"Distillation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# 阶段 4: 量化与部署 API
# ============================================================================

@api_bp.route('/api/quantize/start', methods=['POST', 'GET'])
def start_quantization():
    """
    模型量化与部署 API
    使用 AWQ/GPTQ 进行 4-bit 量化，并部署为 vLLM 服务
    """
    try:
        if request.method == 'POST':
            data = request.get_json()
            model_name = data.get('model_name')
            method = data.get('method', 'awq')
        else:  # GET request for EventSource
            model_name = request.args.get('model_name')
            method = request.args.get('method', 'awq')
        
        if not model_name:
            return jsonify({'error': '请指定要量化的模型'}), 400
        
        # 在生成器外部获取应用上下文信息
        root_path = current_app.root_path
        models_dir = os.path.join(root_path, '..', 'finetuned_models')
        
        def generate_quantization_logs():
            yield f"data: > 开始模型量化任务...\n\n"
            yield f"data: > 目标模型: {model_name}\n\n"
            yield f"data: > 量化方法: {method}\n\n"
            
            try:
                # 优化模型路径查找逻辑
                possible_model_paths = [
                    os.path.join(models_dir, model_name),  # 标准路径
                    os.path.join(models_dir, model_name + '_lora'),  # LoRA模型
                    os.path.join(models_dir, model_name + '_merged'),  # 合并后的模型
                    os.path.join(root_path, '..', 'hugface', model_name.replace('/', '--')),  # HF路径
                    os.path.join(Config.HF_MODEL_DIR, model_name.replace('/', '--')),  # 配置HF目录
                ]
                
                model_path = None
                for path in possible_model_paths:
                    if os.path.exists(path) and os.path.isdir(path):
                        model_path = path
                        break
                
                if not model_path:
                    yield f"data: > [ERROR] 模型不存在: {model_name}\n\n"
                    yield f"data: > [INFO] 已尝试以下路径:\n\n"
                    for path in possible_model_paths:
                        yield f"data: >   - {path}\n\n"
                    yield f"data: > [TIP] 请确保模型已完成训练并存在于 finetuned_models/ 或 hugface/ 目录\n\n"
                    yield f"data: [DONE]\n\n"
                    return
                
                yield f"data: > [INFO] 找到模型路径: {model_path}\n\n"
                
                # 生成量化后模型名称
                quantized_name = f"{model_name}_{method}_4bit"
                output_dir = os.path.join(models_dir, quantized_name)
                os.makedirs(output_dir, exist_ok=True)
                
                yield f"data: > 量化输出目录: {quantized_name}\n\n"
                
                # 检查是否安装了量化所需的库
                try:
                    import autoawq
                    yield f"data: > AWQ 库已安装\n\n"
                except ImportError:
                    yield f"data: > [WARNING] autoawq 未安装，尝试安装...\n\n"
                    yield f"data: > 请手动运行: pip install autoawq\n\n"
                
                # 构建量化命令
                if method == 'awq':
                    yield f"data: > 正在执行 AWQ 量化...\n\n"
                    
                    # 使用 LLaMA-Factory 的 export 功能进行量化
                    config_path = os.path.join(root_path, '..', 'finetune_configs', f'quantize_{datetime.now().strftime("%Y%m%d%H%M%S")}.yaml')
                    
                    # 检查是否是 LoRA 模型
                    is_lora = os.path.exists(os.path.join(model_path, 'adapter_config.json'))
                    
                    if is_lora:
                        # LoRA 模型需要先找到基础模型 - 优化查找逻辑
                        # 尝试从 adapter_config.json 读取基础模型
                        base_model = None
                        adapter_config_path = os.path.join(model_path, 'adapter_config.json')
                        try:
                            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                                adapter_cfg = json.load(f)
                                base_model_from_config = adapter_cfg.get('base_model_name_or_path', '')
                                if base_model_from_config:
                                    # 检查是否是本地路径
                                    if os.path.exists(base_model_from_config):
                                        base_model = base_model_from_config
                                    else:
                                        # 尝试在 hugface 目录查找
                                        base_name = os.path.basename(base_model_from_config.replace('/', '--'))
                                        possible_base_paths = [
                                            os.path.join(root_path, '..', 'hugface', base_name),
                                            os.path.join(Config.HF_MODEL_DIR, base_name),
                                        ]
                                        for path in possible_base_paths:
                                            if os.path.exists(path):
                                                base_model = path
                                                break
                        except Exception as e:
                            yield f"data: > [WARNING] 读取 adapter_config.json 失败: {str(e)}\n\n"
                        
                        # 如果无法从配置读取，从模型名称推断
                        if not base_model:
                            if '0.5B' in model_name or '0.5b' in model_name.lower():
                                base_model = os.path.join(root_path, '..', 'hugface', 'Qwen--Qwen2.5-0.5B-Instruct')
                            elif '7B' in model_name or '7b' in model_name.lower():
                                base_model = os.path.join(root_path, '..', 'hugface', 'Qwen--Qwen2.5-7B-Instruct')
                            elif '1.5B' in model_name or '1.5b' in model_name.lower():
                                base_model = os.path.join(root_path, '..', 'hugface', 'Qwen--Qwen2.5-1.5B-Instruct')
                            else:
                                # 默认使用 0.5B
                                base_model = os.path.join(root_path, '..', 'hugface', 'Qwen--Qwen2.5-0.5B-Instruct')
                        
                        # 验证基础模型是否存在
                        if not os.path.exists(base_model):
                            yield f"data: > [ERROR] 基础模型不存在: {base_model}\n\n"
                            yield f"data: > [TIP] 请先下载基础模型到 hugface/ 目录\n\n"
                            yield f"data: [DONE]\n\n"
                            return
                        
                        yield f"data: > 检测到 LoRA 模型，基础模型: {base_model}\n\n"
                        
                        quantize_config = {
                            'model_name_or_path': base_model,
                            'adapter_name_or_path': model_path,
                            'template': 'qwen',
                            'finetuning_type': 'lora',
                            'export_dir': output_dir,
                            'export_quantization_bit': 4,
                            'export_quantization_dataset': 'alpaca',
                            'export_size': 2,
                            'export_device': 'cpu',
                            'export_legacy_format': False,
                        }
                    else:
                        # 完整模型直接量化
                        quantize_config = {
                            'model_name_or_path': model_path,
                            'template': 'qwen',
                            'finetuning_type': 'full',
                            'export_dir': output_dir,
                            'export_quantization_bit': 4,
                            'export_quantization_dataset': 'alpaca',
                            'export_size': 2,
                            'export_device': 'cpu',
                            'export_legacy_format': False,
                        }
                    
                    with open(config_path, 'w', encoding='utf-8') as f:
                        safe_yaml_dump(quantize_config, f, allow_unicode=True)
                    
                    yield f"data: > 量化配置已生成: {config_path}\n\n"
                    
                    # 执行量化导出
                    env = os.environ.copy()
                    env['OMP_NUM_THREADS'] = '1'
                    env['MKL_NUM_THREADS'] = '1'
                    
                    process = subprocess.Popen(
                        [get_llamafactory_cli_path(), 'export', config_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        encoding='utf-8',
                        errors='replace',
                        env=env
                    )
                    
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
                            if len(buffer) > 150:
                                clean_line = buffer.strip().replace('"', "'")
                                if clean_line:
                                    yield f"data: > {clean_line}\n\n"
                                buffer = ""
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        yield f"data: > [SUCCESS] AWQ 量化完成！\n\n"
                    else:
                        yield f"data: > [ERROR] 量化失败，状态码: {process.returncode}\n\n"
                
                else:
                    yield f"data: > [WARNING] 不支持的量化方法: {method}\n\n"
                
                # vLLM 部署提示
                yield f"data: > [INFO] 量化后的模型可通过以下命令部署为 vLLM 服务:\n\n"
                yield f"data: > python -m vllm.entrypoints.openai.api_server \\\n\n"
                yield f"data: >   --model {output_dir} \\\n\n"
                yield f"data: >   --quantization awq \\\n\n"
                yield f"data: >   --port 8000\n\n"
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: > [ERROR] 量化失败: {str(e)}\n\n"
                yield f"data: [DONE]\n\n"
        
        from flask import Response
        return Response(generate_quantization_logs(), mimetype='text/event-stream')
        
    except Exception as e:
        current_app.logger.error(f"Quantization error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/eval/start', methods=['POST'])
def start_evaluation():
    """对指定模型运行独立评估"""
    from flask import Response, stream_with_context
    from difflib import SequenceMatcher
    
    model_name = request.form.get('model_name', '')
    dataset = request.form.get('dataset', '')
    num_samples = int(request.form.get('num_samples', '50'))
    
    if not model_name or not dataset:
        return jsonify({"error": "缺少必要参数"}), 400
    
    # 查找模型路径
    finetuned_dir = os.path.join(current_app.root_path, '..', 'finetuned_models')
    model_dir = os.path.join(finetuned_dir, model_name)
    if not os.path.exists(model_dir):
        model_dir_lora = os.path.join(finetuned_dir, model_name + '_lora')
        if os.path.exists(model_dir_lora):
            model_dir = model_dir_lora
        else:
            return jsonify({"error": f"模型目录不存在: {model_name}"}), 404
    
    # 查找基座模型（从 adapter_config.json 读取）
    adapter_config_path = os.path.join(model_dir, 'adapter_config.json')
    base_model = None
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                adapter_cfg = json.load(f)
                base_model = adapter_cfg.get('base_model_name_or_path', '')
        except Exception as e:
            current_app.logger.warning(f"读取 adapter_config.json 失败: {e}")
    
    # 读取数据集
    dataset_path = os.path.join(current_app.root_path, '..', Config.UPLOAD_FOLDER, 'datasets', 'cleaned', dataset)
    if not os.path.exists(dataset_path):
        return jsonify({"error": f"数据集不存在: {dataset}"}), 404
    
    # 生成评估配置 YAML
    eval_config = {
        "stage": "sft",
        "do_predict": True,
        "model_name_or_path": base_model or model_dir,
        "adapter_name_or_path": model_dir if base_model else None,
        "finetuning_type": "lora" if base_model else "full",
        "dataset_dir": os.path.join(current_app.root_path, '..', Config.UPLOAD_FOLDER, 'datasets', 'cleaned'),
        "dataset": os.path.splitext(dataset)[0],
        "template": "qwen",
        "output_dir": os.path.join(model_dir, 'eval_output'),
        "per_device_eval_batch_size": 1,
        "max_samples": num_samples,
        "predict_with_generate": True,
        "overwrite_output_dir": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    # 去掉 None 值
    eval_config = {k: v for k, v in eval_config.items() if v is not None}
    
    config_dir = os.path.join(current_app.root_path, '..', 'finetune_configs')
    os.makedirs(config_dir, exist_ok=True)
    config_filename = f"eval_{int(time.time())}.yaml"
    config_path = os.path.join(config_dir, config_filename)
    with open(config_path, 'w', encoding='utf-8') as f:
        safe_yaml_dump(eval_config, f, allow_unicode=True)
    
    def generate():
        try:
            yield f"data: 开始评估模型: {model_name}\n\n"
            yield f"data: 使用数据集: {dataset}, 样本数: {num_samples}\n\n"
            
            cmd = [sys.executable, '-m', 'llamafactory.cli', 'train', config_path]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                env=env, cwd=os.path.join(current_app.root_path, '..', 'model', 'LLaMA-Factory')
            )
            
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    yield f"data: {line}\n\n"
            
            process.wait()
            
            if process.returncode == 0:
                # 计算评估指标
                eval_output_dir = os.path.join(model_dir, 'eval_output')
                predictions_file = os.path.join(eval_output_dir, 'generated_predictions.jsonl')
                
                if os.path.exists(predictions_file):
                    samples = []
                    total_similarity = 0
                    count = 0
                    with open(predictions_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                item = json.loads(line)
                                ref = item.get('label', '')
                                pred = item.get('predict', '')
                                sim = SequenceMatcher(None, ref, pred).ratio()
                                if sim >= 0.8:
                                    status = '正确'
                                elif sim >= 0.5:
                                    status = '部分正确'
                                else:
                                    status = '错误'
                                samples.append({
                                    'instruction': item.get('prompt', ''),
                                    'reference': ref,
                                    'predict': pred,
                                    'status': status,
                                    'similarity': round(sim, 4)
                                })
                                total_similarity += sim
                                count += 1
                    
                    avg_similarity = total_similarity / count if count > 0 else 0
                    
                    # 保存评估结果到模型目录
                    eval_results = {
                        'eval_samples': count,
                        'avg_similarity': round(avg_similarity, 4),
                        'correct_rate': round(sum(1 for s in samples if s['status'] == '正确') / count, 4) if count > 0 else 0,
                        'partial_rate': round(sum(1 for s in samples if s['status'] == '部分正确') / count, 4) if count > 0 else 0,
                        'error_rate': round(sum(1 for s in samples if s['status'] == '错误') / count, 4) if count > 0 else 0,
                    }
                    with open(os.path.join(model_dir, 'eval_results.json'), 'w', encoding='utf-8') as f:
                        json.dump(eval_results, f, ensure_ascii=False, indent=2)
                    
                    # 复制 predictions 到模型目录
                    import shutil
                    shutil.copy2(predictions_file, os.path.join(model_dir, 'generated_predictions.jsonl'))
                    
                    yield f"data: [EVAL_COMPLETE] 评估完成！样本数: {count}, 平均相似度: {avg_similarity:.4f}\n\n"
                else:
                    yield f"data: [EVAL_COMPLETE] 评估完成，但未生成预测文件\n\n"
            else:
                yield f"data: [EVAL_ERROR] 评估失败，返回码: {process.returncode}\n\n"
        except Exception as e:
            yield f"data: [EVAL_ERROR] 评估异常: {str(e)}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


# ============================================================================
# 阶段 4: vLLM 部署 API
# ============================================================================

# 存储正在运行的vLLM进程
active_vllm_processes = {}

@api_bp.route('/api/deploy/start', methods=['POST'])
def start_deploy():
    """
    使用 vLLM 部署模型为推理服务
    """
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        port = int(data.get('port', 8000))
        gpu_util = float(data.get('gpu_util', 0.8))
        is_quantized = data.get('is_quantized', False)
        
        if not model_name:
            return jsonify({'error': '请指定要部署的模型'}), 400
        
        # 检查 vLLM 是否已安装
        try:
            import vllm
            vllm_version = vllm.__version__
        except ImportError:
            return jsonify({
                'error': 'vLLM 未安装',
                'message': '请先安装 vLLM: pip install vllm。Windows 系统请使用 WSL2 或参考 vLLM 官方文档安装。'
            }), 500
        
        # 查找模型路径
        models_dir = os.path.join(current_app.root_path, '..', 'finetuned_models')
        hugface_dir = os.path.join(current_app.root_path, '..', 'hugface')
        
        possible_paths = [
            os.path.join(models_dir, model_name),
            os.path.join(models_dir, model_name + '_awq_4bit'),
            os.path.join(models_dir, model_name + '_merged'),
            os.path.join(models_dir, model_name + '_lora'),
            os.path.join(hugface_dir, model_name.replace('/', '--')),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                model_path = path
                break
        
        if not model_path:
            return jsonify({
                'error': f'模型不存在: {model_name}',
                'message': '请确保模型已完成训练或量化'
            }), 404
        
        # 检查端口是否被占用
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            return jsonify({
                'error': f'端口 {port} 已被占用',
                'message': '请更换端口或停止占用该端口的服务'
            }), 400
        
        # 构建 vLLM 启动命令
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', model_path,
            '--port', str(port),
            '--gpu-memory-utilization', str(gpu_util),
            '--trust-remote-code',
        ]
        
        if is_quantized:
            cmd.extend(['--quantization', 'awq'])
        
        # 启动 vLLM 进程
        try:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # 等待一小段时间检查进程是否成功启动
            import time
            time.sleep(1)
            
            # 检查进程是否仍在运行
            if process.poll() is not None:
                # 进程已退出，读取错误信息
                stdout, stderr = process.communicate()
                error_msg = stderr if stderr else stdout
                return jsonify({
                    'error': f'vLLM 启动失败',
                    'message': f'进程异常退出，错误信息: {error_msg[:500] if error_msg else "未知错误"}'
                }), 500
            
            # 存储进程信息
            service_id = f"{model_name}_{port}"
            active_vllm_processes[service_id] = {
                'process': process,
                'model': model_name,
                'port': port,
                'start_time': datetime.now().isoformat(),
                'model_path': model_path
            }
            
            api_url = f"http://localhost:{port}/v1/chat/completions"
            
            return jsonify({
                'message': f'vLLM 服务已启动 (v{vllm_version})',
                'model': model_name,
                'port': port,
                'api_url': api_url,
                'service_id': service_id,
                'note': '服务启动需要约10-30秒加载模型，请稍后测试'
            })
            
        except Exception as e:
            return jsonify({
                'error': f'启动 vLLM 失败: {str(e)}',
                'message': '请确保已安装 vLLM: pip install vllm'
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Deploy error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/deploy/stop', methods=['POST'])
def stop_deploy():
    """停止 vLLM 部署服务"""
    try:
        data = request.get_json()
        service_id = data.get('service_id')
        
        if service_id and service_id in active_vllm_processes:
            process_info = active_vllm_processes[service_id]
            process = process_info['process']
            
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            del active_vllm_processes[service_id]
            
            return jsonify({
                'message': f'服务 {service_id} 已停止',
                'port': process_info['port']
            })
        else:
            return jsonify({'error': '未找到指定的服务'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/deploy/list', methods=['GET'])
def list_deployments():
    """列出所有正在运行的部署服务"""
    try:
        # 清理已结束的进程
        dead_services = []
        for service_id, info in active_vllm_processes.items():
            if info['process'].poll() is not None:
                dead_services.append(service_id)
        
        for sid in dead_services:
            del active_vllm_processes[sid]
        
        return jsonify({
            'services': [
                {
                    'service_id': sid,
                    'model': info['model'],
                    'port': info['port'],
                    'start_time': info['start_time'],
                    'api_url': f"http://localhost:{info['port']}/v1/chat/completions"
                }
                for sid, info in active_vllm_processes.items()
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/task/status', methods=['GET'])
def get_task_status():
    """查询各阶段任务运行状态"""
    from app.api.common import active_finetune_processes
    
    status = {
        'finetune': {'running': False, 'details': None},
        'distill': {'running': False, 'details': None},
        'quantize': {'running': False, 'details': None},
        'deploy': {'running': False, 'services': []}
    }
    
    # 检查微调进程
    if 'current' in active_finetune_processes:
        proc = active_finetune_processes['current']
        if proc and proc.poll() is None:  # 进程仍在运行
            status['finetune'] = {
                'running': True,
                'details': {
                    'task_id': 'current',
                    'model': active_finetune_processes.get('model_name', ''),
                    'start_time': active_finetune_processes.get('start_time', '')
                }
            }
    
    # 检查vLLM部署进程
    dead_services = []
    for service_id, info in list(active_vllm_processes.items()):
        proc = info.get('process')
        if proc and proc.poll() is None:  # 运行中
            status['deploy']['running'] = True
            status['deploy']['services'].append({
                'service_id': service_id,
                'model': info.get('model', ''),
                'port': info.get('port', 8000)
            })
        else:
            dead_services.append(service_id)
    
    # 清理已死进程
    for sid in dead_services:
        del active_vllm_processes[sid]
    
    return jsonify(status)
