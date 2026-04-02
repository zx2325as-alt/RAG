"""
模型训练评估API路由
包含训练监控、模型评估、多模型对比功能
"""
import os
import json
import time
from datetime import datetime
from flask import request, jsonify, Response, current_app, Blueprint

# 创建独立的 Blueprint
eval_bp = Blueprint('eval', __name__)

from app.db.models import TrainingJob, ModelEvaluation, ModelComparison, db
from app.services.eval_service import EvalService


# ==================== 模块A：训练过程监控 ====================

@eval_bp.route('/api/training/jobs', methods=['GET'])
def get_training_jobs():
    """获取训练任务列表"""
    try:
        status = request.args.get('status')
        query = TrainingJob.query
        
        if status:
            query = query.filter_by(status=status)
        
        jobs = query.order_by(TrainingJob.created_at.desc()).all()
        
        # 如果没有数据库记录，从文件系统扫描
        if not jobs:
            root_path = current_app.root_path
            models_dir = os.path.join(root_path, '..', 'finetuned_models')
            
            if os.path.exists(models_dir):
                for name in os.listdir(models_dir):
                    model_path = os.path.join(models_dir, name)
                    if os.path.isdir(model_path):
                        # 跳过特殊目录
                        if name in ['runs', 'base_models']:
                            continue
                        # 检查是否有模型文件
                        has_model = (
                            os.path.exists(os.path.join(model_path, 'model.safetensors')) or
                            os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) or
                            os.path.exists(os.path.join(model_path, 'adapter_model.safetensors')) or
                            os.path.exists(os.path.join(model_path, 'adapter_model.bin')) or
                            any(f.startswith('model-') and f.endswith('.safetensors') for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f)))
                        )
                        if has_model:
                            # 检查是否有训练状态文件
                            trainer_state_path = os.path.join(model_path, 'trainer_state.json')
                            status = 'completed' if os.path.exists(trainer_state_path) else 'unknown'
                            jobs.append({
                                'id': None,
                                'model_name': name,
                                'status': status,
                                'start_time': None,
                                'end_time': None,
                                'metrics_summary': {}
                            })
        
        return jsonify({
            'jobs': [{
                'id': job.id if hasattr(job, 'id') else job['id'],
                'model_name': job.model_name if hasattr(job, 'model_name') else job['model_name'],
                'status': job.status if hasattr(job, 'status') else job['status'],
                'start_time': job.start_time.isoformat() if hasattr(job, 'start_time') and job.start_time else (job.get('start_time') if isinstance(job, dict) else None),
                'end_time': job.end_time.isoformat() if hasattr(job, 'end_time') and job.end_time else (job.get('end_time') if isinstance(job, dict) else None),
                'metrics_summary': job.metrics_summary if hasattr(job, 'metrics_summary') else job.get('metrics_summary', {})
            } for job in jobs]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@eval_bp.route('/api/training/<model_name>/status', methods=['GET'])
def get_training_status(model_name):
    """获取训练实时状态"""
    try:
        # 从trainer_state.json读取实时状态
        root_path = current_app.root_path
        model_path = os.path.join(root_path, '..', 'finetuned_models', model_name)
        trainer_state_path = os.path.join(model_path, 'trainer_state.json')
        
        if not os.path.exists(trainer_state_path):
            return jsonify({'error': 'Model not found or not trained'}), 404
        
        trainer_state = EvalService.load_trainer_state(model_path)
        metrics = EvalService.extract_training_metrics(trainer_state)
        
        current_app.logger.info(f"Status API - Model: {model_name}, latest_loss: {metrics.get('final_loss')}, current_step: {metrics.get('current_step')}, total_steps: {metrics.get('total_steps')}")
        
        # 判断训练状态
        is_training = False
        if metrics.get('current_step', 0) < metrics.get('total_steps', 0):
            # 检查是否有最近更新（5分钟内）
            trainer_log_path = os.path.join(model_path, 'trainer_log.jsonl')
            if os.path.exists(trainer_log_path):
                mtime = os.path.getmtime(trainer_log_path)
                is_training = (time.time() - mtime) < 300  # 5分钟
        
        # 计算预计剩余时间
        estimated_remaining = None
        if is_training and metrics.get('current_step', 0) > 0:
            # 从数据库获取开始时间
            job = TrainingJob.query.filter_by(model_name=model_name).first()
            if job and job.start_time:
                elapsed = (datetime.utcnow() - job.start_time).total_seconds()
                steps_done = metrics['current_step']
                steps_total = metrics['total_steps']
                if steps_done > 0:
                    time_per_step = elapsed / steps_done
                    remaining_steps = steps_total - steps_done
                    remaining_seconds = time_per_step * remaining_steps
                    estimated_remaining = f"{int(remaining_seconds // 3600)}h {int((remaining_seconds % 3600) // 60)}m"
        
        return jsonify({
            'model_name': model_name,
            'is_training': is_training,
            'current_step': metrics.get('current_step', 0),
            'total_steps': metrics.get('total_steps', 0),
            'current_epoch': metrics.get('current_step', 0) / max(metrics.get('total_steps', 1), 1) * 3,  # 假设3个epoch
            'latest_loss': metrics.get('final_loss'),
            'latest_eval_loss': metrics.get('final_eval_loss'),
            'estimated_remaining_time': estimated_remaining,
            'best_metric': metrics.get('best_metric')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@eval_bp.route('/api/training/<model_name>/metrics', methods=['GET'])
def get_training_metrics(model_name):
    """获取训练历史指标（用于图表）"""
    try:
        root_path = current_app.root_path
        model_path = os.path.join(root_path, '..', 'finetuned_models', model_name)
        
        trainer_state = EvalService.load_trainer_state(model_path)
        metrics = EvalService.extract_training_metrics(trainer_state)
        
        current_app.logger.info(f"Model: {model_name}, log_history count: {len(trainer_state.get('log_history', []))}")
        current_app.logger.info(f"loss_history count: {len(metrics.get('loss_history', []))}")
        
        return jsonify({
            'model_name': model_name,
            'loss_history': metrics.get('loss_history', []),
            'lr_history': metrics.get('lr_history', []),
            'grad_norm_history': metrics.get('grad_norm_history', []),
            'eval_loss_history': metrics.get('eval_loss_history', [])
        })
        
    except Exception as e:
        current_app.logger.error(f"get_training_metrics error: {e}")
        return jsonify({'error': str(e)}), 500


@eval_bp.route('/api/training/<model_name>/stream', methods=['GET'])
def stream_training_logs(model_name):
    """SSE流式推送训练日志"""
    def generate():
        last_step = 0
        while True:
            try:
                root_path = current_app.root_path
                model_path = os.path.join(root_path, '..', 'finetuned_models', model_name)
                trainer_state = EvalService.load_trainer_state(model_path)
                metrics = EvalService.extract_training_metrics(trainer_state)
                
                current_step = metrics.get('current_step', 0)
                
                # 只有当有更新时才推送
                if current_step > last_step:
                    last_step = current_step
                    data = {
                        'step': current_step,
                        'total_steps': metrics.get('total_steps', 0),
                        'loss': metrics.get('final_loss'),
                        'eval_loss': metrics.get('final_eval_loss'),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                
                # 检查训练是否完成
                if current_step >= metrics.get('total_steps', 0) and current_step > 0:
                    yield f"data: {json.dumps({'status': 'completed'})}\n\n"
                    break
                    
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            time.sleep(5)  # 每5秒检查一次
    
    return Response(generate(), mimetype='text/event-stream')


# ==================== 模块B：模型评估 ====================

@eval_bp.route('/api/eval/models', methods=['GET'])
def get_evaluable_models():
    """获取可评估的模型列表"""
    try:
        root_path = current_app.root_path
        models_dir = os.path.join(root_path, '..', 'finetuned_models')
        
        models = []
        if os.path.exists(models_dir):
            for name in os.listdir(models_dir):
                model_path = os.path.join(models_dir, name)
                if os.path.isdir(model_path):
                    # 跳过特殊目录
                    if name in ['runs', 'base_models']:
                        continue
                    # 检查是否有模型文件（包括完整模型和LoRA模型）
                    has_model = (
                        os.path.exists(os.path.join(model_path, 'model.safetensors')) or
                        os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) or
                        os.path.exists(os.path.join(model_path, 'adapter_model.safetensors')) or
                        os.path.exists(os.path.join(model_path, 'adapter_model.bin')) or
                        os.path.exists(os.path.join(model_path, 'model-00001-of-00001.safetensors')) or
                        any(f.startswith('model-') and f.endswith('.safetensors') for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f)))
                    )
                    if has_model:
                        models.append(name)
        
        current_app.logger.info(f"Found {len(models)} models: {models}")
        return jsonify({'models': sorted(models)})
        
    except Exception as e:
        current_app.logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500


@eval_bp.route('/api/eval/datasets', methods=['GET'])
def get_eval_datasets():
    """获取评估数据集列表"""
    try:
        root_path = current_app.root_path
        datasets_dir = os.path.join(root_path, '..', 'uploads', 'datasets', 'cleaned')
        
        datasets = []
        if os.path.exists(datasets_dir):
            for f in os.listdir(datasets_dir):
                if f.endswith('.jsonl') or f.endswith('.json'):
                    datasets.append(f)
        else:
            # 如果cleaned目录不存在，检查raw目录
            raw_dir = os.path.join(root_path, '..', 'uploads', 'datasets', 'raw')
            if os.path.exists(raw_dir):
                for f in os.listdir(raw_dir):
                    if f.endswith('.jsonl') or f.endswith('.json'):
                        datasets.append(f)
        
        current_app.logger.info(f"Found {len(datasets)} datasets: {datasets}")
        return jsonify({'datasets': sorted(datasets)})
        
    except Exception as e:
        current_app.logger.error(f"Error getting datasets: {e}")
        return jsonify({'error': str(e)}), 500


@eval_bp.route('/api/eval/model/start', methods=['POST'])
def start_model_eval():
    """启动模型评估"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        eval_dataset = data.get('eval_dataset')
        sample_count = data.get('sample_count', 50)
        
        if not model_name or not eval_dataset:
            return jsonify({'error': '请指定模型和数据集'}), 400
        
        # 检查模型是否存在
        root_path = current_app.root_path
        model_path = os.path.join(root_path, '..', 'finetuned_models', model_name)
        if not os.path.exists(model_path):
            return jsonify({'error': '模型不存在'}), 404
        
        # 检查数据集是否存在
        dataset_path = os.path.join(root_path, '..', 'uploads', 'datasets', 'cleaned', eval_dataset)
        if not os.path.exists(dataset_path):
            return jsonify({'error': '数据集不存在'}), 404
        
        def generate_eval_logs():
            yield f"data: > 开始评估模型: {model_name}\n\n"
            yield f"data: > 数据集: {eval_dataset}\n\n"
            yield f"data: > 样本数: {sample_count}\n\n"
            
            try:
                def progress_callback(msg, progress):
                    yield f"data: > [{progress}%] {msg}\n\n"
                
                metrics, sample_comparisons = EvalService.evaluate_model_on_dataset(
                    model_path, dataset_path, sample_count, progress_callback
                )
                
                # 保存评估结果到数据库
                eval_record = ModelEvaluation(
                    model_name=model_name,
                    eval_dataset=eval_dataset,
                    sample_count=sample_count,
                    metrics=metrics,
                    sample_comparisons=sample_comparisons[:10]  # 只保存前10个样例
                )
                db.session.add(eval_record)
                db.session.commit()
                
                yield f"data: > 评估完成!\n\n"
                yield f"data: > ROUGE-L: {metrics.get('rougeL', 0)}\n\n"
                yield f"data: > BLEU: {metrics.get('bleu', 0)}\n\n"
                yield f"data: > BERTScore F1: {metrics.get('bertscore_f1', 0)}\n\n"
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: > [ERROR] 评估失败: {str(e)}\n\n"
                yield f"data: [DONE]\n\n"
        
        return Response(
            generate_eval_logs(),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@eval_bp.route('/api/eval/model/results', methods=['GET'])
def get_model_eval_results():
    """获取模型评估结果列表"""
    try:
        model_name = request.args.get('model_name')
        limit = request.args.get('limit', 20, type=int)
        
        query = ModelEvaluation.query
        if model_name:
            query = query.filter_by(model_name=model_name)
        
        results = query.order_by(ModelEvaluation.created_at.desc()).limit(limit).all()
        
        return jsonify({
            'results': [{
                'id': r.id,
                'model_name': r.model_name,
                'eval_dataset': r.eval_dataset,
                'sample_count': r.sample_count,
                'metrics': r.metrics,
                'created_at': r.created_at.isoformat()
            } for r in results]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@eval_bp.route('/api/eval/model/result/<int:eval_id>', methods=['GET'])
def get_model_eval_detail(eval_id):
    """获取单个评估结果的详细信息"""
    try:
        eval_record = ModelEvaluation.query.get_or_404(eval_id)
        
        return jsonify({
            'id': eval_record.id,
            'model_name': eval_record.model_name,
            'eval_dataset': eval_record.eval_dataset,
            'sample_count': eval_record.sample_count,
            'metrics': eval_record.metrics,
            'efficiency_metrics': eval_record.efficiency_metrics,
            'sample_comparisons': eval_record.sample_comparisons,
            'created_at': eval_record.created_at.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== 模块C：多模型对比 ====================

@eval_bp.route('/api/eval/comparison/start', methods=['POST'])
def start_model_comparison():
    """启动多模型对比评估"""
    try:
        data = request.get_json()
        comparison_name = data.get('comparison_name', '模型对比')
        model_names = data.get('model_names', [])
        eval_dataset = data.get('eval_dataset')
        sample_count = data.get('sample_count', 30)
        
        if len(model_names) < 2:
            return jsonify({'error': '请至少选择2个模型进行对比'}), 400
        
        if len(model_names) > 4:
            return jsonify({'error': '最多只能选择4个模型进行对比'}), 400
        
        if not eval_dataset:
            return jsonify({'error': '请指定评估数据集'}), 400
        
        # 检查数据集
        root_path = current_app.root_path
        dataset_path = os.path.join(root_path, '..', 'uploads', 'datasets', 'cleaned', eval_dataset)
        if not os.path.exists(dataset_path):
            return jsonify({'error': '数据集不存在'}), 404
        
        def generate_comparison_logs():
            yield f"data: > 开始多模型对比评估: {comparison_name}\n\n"
            yield f"data: > 对比模型: {', '.join(model_names)}\n\n"
            yield f"data: > 数据集: {eval_dataset}\n\n"
            
            try:
                def progress_callback(msg, progress):
                    yield f"data: > [{progress}%] {msg}\n\n"
                
                results = EvalService.compare_models(
                    model_names, dataset_path, sample_count, progress_callback
                )
                
                # 保存对比结果到数据库
                comparison = ModelComparison(
                    comparison_name=comparison_name,
                    model_names=model_names,
                    eval_dataset=eval_dataset,
                    model_metrics=results['model_metrics'],
                    sample_comparisons=results['sample_comparisons'][:5]  # 只保存前5个样例
                )
                db.session.add(comparison)
                db.session.commit()
                
                # 输出对比结果摘要
                yield f"data: > 对比完成!\n\n"
                yield f"data: > 结果摘要:\n\n"
                
                for model_name, metrics in results['model_metrics'].items():
                    if 'error' not in metrics:
                        yield f"data: >   {model_name}:\n\n"
                        yield f"data: >     ROUGE-L: {metrics.get('rougeL', 0)}\n\n"
                        yield f"data: >     BLEU: {metrics.get('bleu', 0)}\n\n"
                
                yield f"data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: > [ERROR] 对比失败: {str(e)}\n\n"
                yield f"data: [DONE]\n\n"
        
        return Response(
            generate_comparison_logs(),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@eval_bp.route('/api/eval/comparison/list', methods=['GET'])
def get_comparison_list():
    """获取历史对比记录"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        comparisons = ModelComparison.query.order_by(
            ModelComparison.created_at.desc()
        ).limit(limit).all()
        
        return jsonify({
            'comparisons': [{
                'id': c.id,
                'comparison_name': c.comparison_name,
                'model_names': c.model_names,
                'eval_dataset': c.eval_dataset,
                'model_metrics': c.model_metrics,
                'created_at': c.created_at.isoformat()
            } for c in comparisons]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@eval_bp.route('/api/eval/comparison/<int:comparison_id>', methods=['GET'])
def get_comparison_detail(comparison_id):
    """获取对比详情"""
    try:
        comparison = ModelComparison.query.get_or_404(comparison_id)
        
        return jsonify({
            'id': comparison.id,
            'comparison_name': comparison.comparison_name,
            'model_names': comparison.model_names,
            'eval_dataset': comparison.eval_dataset,
            'model_metrics': comparison.model_metrics,
            'sample_comparisons': comparison.sample_comparisons,
            'created_at': comparison.created_at.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@eval_bp.route('/api/eval/comparison/<int:comparison_id>/export', methods=['GET'])
def export_comparison_report(comparison_id):
    """导出对比报告为HTML"""
    try:
        comparison = ModelComparison.query.get_or_404(comparison_id)
        
        # 生成HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>模型对比报告 - {comparison.comparison_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #1e3c72; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #1e3c72; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #2a5298; }}
            </style>
        </head>
        <body>
            <h1>模型对比报告: {comparison.comparison_name}</h1>
            <p>生成时间: {comparison.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>评估数据集: {comparison.eval_dataset}</p>
            
            <h2>指标对比</h2>
            <table>
                <tr>
                    <th>模型</th>
                    <th>ROUGE-L</th>
                    <th>BLEU</th>
                    <th>BERTScore F1</th>
                </tr>
        """
        
        for model_name, metrics in comparison.model_metrics.items():
            if 'error' not in metrics:
                html_content += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{metrics.get('rougeL', 'N/A')}</td>
                    <td>{metrics.get('bleu', 'N/A')}</td>
                    <td>{metrics.get('bertscore_f1', 'N/A')}</td>
                </tr>
                """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        return Response(
            html_content,
            mimetype='text/html',
            headers={
                'Content-Disposition': f'attachment; filename=comparison_{comparison_id}.html'
            }
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
