"""
模型缓存管理模块
实现模型单例模式，避免重复加载，减少内存占用
"""
import os
import sys
import torch
from typing import Dict, Any, Optional
from threading import Lock
from functools import lru_cache
import hashlib
import json

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config import Config


class ModelCache:
    """
    模型缓存管理器 - 单例模式
    用于缓存加载的模型，避免重复加载造成的内存浪费
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            self._cache: Dict[str, Any] = {}
            self._metadata: Dict[str, Dict] = {}
            self._access_count: Dict[str, int] = {}
            self._initialized = True
            
            print("[ModelCache] 模型缓存管理器初始化完成")
    
    def _get_cache_key(self, model_type: str, model_name: str, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            model_type: 模型类型 (embedding, llm, reranker, etc.)
            model_name: 模型名称或路径
            **kwargs: 其他影响模型加载的参数
        
        Returns:
            缓存键字符串
        """
        # 构建缓存键的基础部分
        key_parts = [model_type, model_name]
        
        # 添加影响模型加载的关键参数
        important_params = ['device', 'torch_dtype', 'quantization_config']
        for param in important_params:
            if param in kwargs:
                key_parts.append(f"{param}={kwargs[param]}")
        
        key_string = "|".join(key_parts)
        
        # 使用 MD5 生成固定长度的键
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_model(self, model_type: str, model_name: str, loader_func, **kwargs) -> Any:
        """
        获取模型，如果缓存中存在则返回缓存的模型，否则加载并缓存
        
        Args:
            model_type: 模型类型
            model_name: 模型名称
            loader_func: 加载模型的函数
            **kwargs: 传递给 loader_func 的参数
        
        Returns:
            加载或缓存的模型实例
        """
        cache_key = self._get_cache_key(model_type, model_name, **kwargs)
        
        with self._lock:
            if cache_key in self._cache:
                self._access_count[cache_key] += 1
                print(f"[ModelCache] Cache hit: {model_type}/{model_name} (accessed {self._access_count[cache_key]} times)")
                return self._cache[cache_key]
        
        # 缓存未命中，加载模型
        print(f"[ModelCache] Cache miss: {model_type}/{model_name}, loading...")
        
        try:
            model = loader_func(**kwargs)
            
            with self._lock:
                self._cache[cache_key] = model
                self._metadata[cache_key] = {
                    'type': model_type,
                    'name': model_name,
                    'kwargs': kwargs,
                    'cache_key': cache_key
                }
                self._access_count[cache_key] = 1
            
            print(f"[ModelCache] Model loaded and cached: {model_type}/{model_name}")
            return model
            
        except Exception as e:
            print(f"[ModelCache] Failed to load model {model_type}/{model_name}: {e}")
            raise
    
    def release_model(self, model_type: str, model_name: str, **kwargs) -> bool:
        """
        释放指定模型的缓存
        
        Args:
            model_type: 模型类型
            model_name: 模型名称
            **kwargs: 模型加载参数
        
        Returns:
            是否成功释放
        """
        cache_key = self._get_cache_key(model_type, model_name, **kwargs)
        
        with self._lock:
            if cache_key in self._cache:
                model = self._cache[cache_key]
                
                # 尝试释放模型资源
                try:
                    if hasattr(model, 'cpu'):
                        model.cpu()
                    del model
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[ModelCache] Warning: Error releasing model: {e}")
                
                del self._cache[cache_key]
                del self._metadata[cache_key]
                del self._access_count[cache_key]
                
                print(f"[ModelCache] Model released: {model_type}/{model_name}")
                return True
            
            return False
    
    def clear_cache(self, model_type: Optional[str] = None) -> int:
        """
        清空缓存
        
        Args:
            model_type: 如果指定，只清空该类型的模型；否则清空所有
        
        Returns:
            释放的模型数量
        """
        with self._lock:
            keys_to_remove = []
            
            for key, metadata in self._metadata.items():
                if model_type is None or metadata.get('type') == model_type:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                if key in self._cache:
                    try:
                        model = self._cache[key]
                        if hasattr(model, 'cpu'):
                            model.cpu()
                        del model
                    except:
                        pass
                    
                    del self._cache[key]
                    del self._metadata[key]
                    del self._access_count[key]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            count = len(keys_to_remove)
            print(f"[ModelCache] Cache cleared: {count} models released")
            return count
    
    def get_cache_info(self) -> Dict:
        """
        获取缓存信息
        
        Returns:
            缓存统计信息
        """
        with self._lock:
            info = {
                'total_models': len(self._cache),
                'models': []
            }
            
            for key, metadata in self._metadata.items():
                info['models'].append({
                    'type': metadata['type'],
                    'name': metadata['name'],
                    'access_count': self._access_count.get(key, 0),
                    'cache_key': key[:8] + '...'  # 只显示部分键
                })
            
            return info
    
    def print_cache_status(self):
        """打印缓存状态"""
        info = self.get_cache_info()
        print("\n" + "="*60)
        print("[ModelCache] 当前缓存状态")
        print("="*60)
        print(f"缓存模型总数: {info['total_models']}")
        print("-"*60)
        
        if info['models']:
            print(f"{'类型':<15} {'名称':<40} {'访问次数':<10}")
            print("-"*60)
            for model in info['models']:
                name = model['name'][:37] + '...' if len(model['name']) > 40 else model['name']
                print(f"{model['type']:<15} {name:<40} {model['access_count']:<10}")
        
        print("="*60 + "\n")


# 全局模型缓存实例
model_cache = ModelCache()


# 便捷函数
def get_cached_embedding_model(model_name: str, **kwargs):
    """获取缓存的 Embedding 模型"""
    from langchain_huggingface import HuggingFaceEmbeddings
    
    def loader(**loader_kwargs):
        return HuggingFaceEmbeddings(**loader_kwargs)
    
    return model_cache.get_model('embedding', model_name, loader, **kwargs)


def get_cached_llm(model_type: str, **kwargs):
    """获取缓存的 LLM 模型"""
    if model_type == 'ollama':
        from langchain_community.chat_models import ChatOllama
        def loader(**loader_kwargs):
            return ChatOllama(**loader_kwargs)
    else:
        from langchain_openai import ChatOpenAI
        def loader(**loader_kwargs):
            return ChatOpenAI(**loader_kwargs)
    
    cache_name = f"{model_type}_{kwargs.get('model', 'default')}"
    return model_cache.get_model('llm', cache_name, loader, **kwargs)


def get_cached_reranker_model(model_name: str, **kwargs):
    """获取缓存的重排序模型"""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    def loader(**loader_kwargs):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **loader_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return {'model': model, 'tokenizer': tokenizer}
    
    return model_cache.get_model('reranker', model_name, loader, **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("测试模型缓存管理器...")
    
    cache = ModelCache()
    
    # 模拟模型加载
    def mock_loader(**kwargs):
        print(f"  正在加载模型，参数: {kwargs}")
        return {"model": "mock_model", "params": kwargs}
    
    # 第一次加载（缓存未命中）
    model1 = cache.get_model("test", "model1", mock_loader, device="cuda")
    
    # 第二次加载（缓存命中）
    model2 = cache.get_model("test", "model1", mock_loader, device="cuda")
    
    # 验证是同一个对象
    print(f"是否是同一个对象: {model1 is model2}")
    
    # 打印缓存状态
    cache.print_cache_status()
    
    # 清空缓存
    cache.clear_cache()
    print("缓存已清空")
