"""
模型评估服务层
提供模型评估、指标计算、对比分析等功能
"""
import os
import json
import time
import torch
from typing import List, Dict, Any, Tuple
from datetime import datetime
from flask import current_app

class EvalService:
    """模型评估服务"""
    
    @staticmethod
    def load_trainer_state(model_path: str) -> Dict[str, Any]:
        """加载trainer_state.json获取训练指标"""
        trainer_state_path = os.path.join(model_path, 'trainer_state.json')
        if not os.path.exists(trainer_state_path):
            return {}
        
        try:
            with open(trainer_state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            current_app.logger.error(f"Failed to load trainer_state: {e}")
            return {}
    
    @staticmethod
    def extract_training_metrics(trainer_state: Dict) -> Dict[str, Any]:
        """从trainer_state提取训练指标"""
        if not trainer_state:
            return {}
        
        log_history = trainer_state.get('log_history', [])
        
        loss_history = []
        lr_history = []
        grad_norm_history = []
        eval_loss_history = []
        
        for log in log_history:
            step = log.get('step', 0)
            
            if 'loss' in log:
                loss_history.append({
                    'step': step,
                    'loss': log['loss'],
                    'epoch': log.get('epoch', 0)
                })
            
            if 'learning_rate' in log:
                lr_history.append({
                    'step': step,
                    'lr': log['learning_rate']
                })
            
            if 'grad_norm' in log:
                grad_norm_history.append({
                    'step': step,
                    'grad_norm': log['grad_norm']
                })
            
            if 'eval_loss' in log:
                eval_loss_history.append({
                    'step': step,
                    'loss': log['eval_loss']
                })
        
        # 获取最终指标
        final_log = log_history[-1] if log_history else {}
        
        return {
            'loss_history': loss_history,
            'lr_history': lr_history,
            'grad_norm_history': grad_norm_history,
            'eval_loss_history': eval_loss_history,
            'final_loss': final_log.get('loss'),
            'final_eval_loss': final_log.get('eval_loss'),
            'total_steps': trainer_state.get('max_steps', 0),
            'current_step': trainer_state.get('global_step', 0),
            'best_metric': trainer_state.get('best_metric', None)
        }
    
    @staticmethod
    def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算ROUGE指标"""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                for key in scores:
                    scores[key].append(score[key].fmeasure)
            
            return {
                'rouge1': sum(scores['rouge1']) / len(scores['rouge1']) if scores['rouge1'] else 0,
                'rouge2': sum(scores['rouge2']) / len(scores['rouge2']) if scores['rouge2'] else 0,
                'rougeL': sum(scores['rougeL']) / len(scores['rougeL']) if scores['rougeL'] else 0
            }
        except ImportError:
            current_app.logger.warning("rouge_score not installed, skipping ROUGE calculation")
            return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        except Exception as e:
            current_app.logger.error(f"ROUGE calculation error: {e}")
            return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    @staticmethod
    def calculate_bleu(predictions: List[str], references: List[str]) -> float:
        """计算BLEU指标"""
        try:
            from sacrebleu import corpus_bleu
            
            # sacrebleu需要list of list作为references
            refs = [[ref] for ref in references]
            bleu = corpus_bleu(predictions, list(zip(*refs)))
            return bleu.score
        except ImportError:
            current_app.logger.warning("sacrebleu not installed, skipping BLEU calculation")
            return 0.0
        except Exception as e:
            current_app.logger.error(f"BLEU calculation error: {e}")
            return 0.0
    
    @staticmethod
    def calculate_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算BERTScore"""
        try:
            from bert_score import score
            
            P, R, F1 = score(predictions, references, lang='zh', verbose=False)
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except ImportError:
            current_app.logger.warning("bert_score not installed, skipping BERTScore calculation")
            return {'precision': 0, 'recall': 0, 'f1': 0}
        except Exception as e:
            current_app.logger.error(f"BERTScore calculation error: {e}")
            return {'precision': 0, 'recall': 0, 'f1': 0}
    
    @staticmethod
    def evaluate_model_on_dataset(
        model_path: str,
        dataset_path: str,
        sample_count: int = 100,
        progress_callback=None
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        在数据集上评估模型
        
        Args:
            model_path: 模型路径
            dataset_path: 数据集路径
            sample_count: 评估样本数
            progress_callback: 进度回调函数
            
        Returns:
            (metrics_dict, sample_comparisons_list)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 加载模型和tokenizer
            if progress_callback:
                progress_callback("正在加载模型...", 5)
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto' if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            model.eval()
            
            if progress_callback:
                progress_callback("模型加载完成", 10)
            
            # 加载数据集
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = [json.loads(line) for line in f if line.strip()]
            
            dataset = dataset[:sample_count]
            
            if progress_callback:
                progress_callback(f"加载数据集: {len(dataset)}条", 15)
            
            # 推理
            predictions = []
            references = []
            sample_comparisons = []
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            for idx, item in enumerate(dataset):
                instruction = item.get('instruction', item.get('prompt', item.get('query', '')))
                input_text = item.get('input', '')
                reference = item.get('output', item.get('response', item.get('answer', '')))
                
                # 构建prompt
                if input_text:
                    prompt = f"{instruction}\n\n{input_text}"
                else:
                    prompt = instruction
                
                # 生成
                inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                if device == 'cuda':
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                predictions.append(prediction)
                references.append(reference)
                
                sample_comparisons.append({
                    'input': prompt,
                    'reference': reference,
                    'prediction': prediction
                })
                
                # 更新进度
                progress = 15 + int((idx + 1) / len(dataset) * 70)
                if progress_callback:
                    progress_callback(f"评估进度: {idx + 1}/{len(dataset)}", progress)
            
            # 计算指标
            if progress_callback:
                progress_callback("计算评估指标...", 90)
            
            rouge_scores = EvalService.calculate_rouge(predictions, references)
            bleu_score = EvalService.calculate_bleu(predictions, references)
            bertscore = EvalService.calculate_bertscore(predictions, references)
            
            metrics = {
                'rouge1': round(rouge_scores['rouge1'], 4),
                'rouge2': round(rouge_scores['rouge2'], 4),
                'rougeL': round(rouge_scores['rougeL'], 4),
                'bleu': round(bleu_score, 4),
                'bertscore_f1': round(bertscore['f1'], 4)
            }
            
            # 为每个样例计算ROUGE分数
            try:
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                for i, sample in enumerate(sample_comparisons):
                    score = scorer.score(sample['reference'], sample['prediction'])
                    sample['rouge_score'] = round(score['rougeL'].fmeasure, 4)
            except:
                for sample in sample_comparisons:
                    sample['rouge_score'] = 0
            
            if progress_callback:
                progress_callback("评估完成", 100)
            
            return metrics, sample_comparisons
            
        except Exception as e:
            current_app.logger.error(f"Model evaluation error: {e}")
            raise
    
    @staticmethod
    def compare_models(
        model_names: List[str],
        dataset_path: str,
        sample_count: int = 50,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        对比多个模型
        
        Returns:
            {
                'model_metrics': {...},
                'sample_comparisons': [...]
            }
        """
        root_path = current_app.root_path
        models_dir = os.path.join(root_path, '..', 'finetuned_models')
        
        model_metrics = {}
        all_sample_outputs = {}
        
        total_models = len(model_names)
        
        for model_idx, model_name in enumerate(model_names):
            model_path = os.path.join(models_dir, model_name)
            
            if not os.path.exists(model_path):
                continue
            
            base_progress = int(model_idx / total_models * 100)
            
            def model_progress(msg, prog):
                if progress_callback:
                    overall_prog = base_progress + int(prog / total_models)
                    progress_callback(f"[{model_name}] {msg}", overall_prog)
            
            try:
                metrics, samples = EvalService.evaluate_model_on_dataset(
                    model_path, dataset_path, sample_count, model_progress
                )
                
                model_metrics[model_name] = metrics
                all_sample_outputs[model_name] = samples
                
            except Exception as e:
                current_app.logger.error(f"Failed to evaluate {model_name}: {e}")
                model_metrics[model_name] = {'error': str(e)}
        
        # 构建样例对比
        sample_comparisons = []
        if all_sample_outputs:
            first_model = list(all_sample_outputs.keys())[0]
            num_samples = len(all_sample_outputs[first_model])
            
            for i in range(num_samples):
                sample_comparison = {
                    'input': all_sample_outputs[first_model][i]['input'],
                    'reference': all_sample_outputs[first_model][i]['reference'],
                    'model_outputs': {}
                }
                
                for model_name in model_names:
                    if model_name in all_sample_outputs and i < len(all_sample_outputs[model_name]):
                        sample = all_sample_outputs[model_name][i]
                        sample_comparison['model_outputs'][model_name] = {
                            'output': sample['prediction'],
                            'score': sample.get('rouge_score', 0)
                        }
                
                sample_comparisons.append(sample_comparison)
        
        return {
            'model_metrics': model_metrics,
            'sample_comparisons': sample_comparisons
        }
