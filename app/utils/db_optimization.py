"""
数据库查询优化工具
提供查询优化、缓存、批处理等功能
"""
import os
import sys
from functools import wraps
from typing import List, Any, Callable
import time
import hashlib
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy.orm import joinedload
from app.db import db


class QueryOptimizer:
    """查询优化器"""
    
    @staticmethod
    def eager_load(relations: List[str]):
        """
        预加载关联数据的装饰器
        
        使用示例:
        @QueryOptimizer.eager_load(['document', 'tags'])
        def get_chunks_with_docs():
            return Chunk.query.all()
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                query = func(*args, **kwargs)
                
                # 应用预加载
                for relation in relations:
                    query = query.options(joinedload(relation))
                
                return query
            return wrapper
        return decorator
    
    @staticmethod
    def batch_query(query, batch_size: int = 100):
        """
        批量查询，避免一次性加载大量数据
        
        Args:
            query: SQLAlchemy 查询对象
            batch_size: 每批大小
        
        Yields:
            每批数据
        """
        offset = 0
        while True:
            batch = query.limit(batch_size).offset(offset).all()
            if not batch:
                break
            yield batch
            offset += batch_size
    
    @staticmethod
    def bulk_insert(model_class, data_list: List[dict], batch_size: int = 1000):
        """
        批量插入数据
        
        Args:
            model_class: 模型类
            data_list: 数据列表
            batch_size: 每批插入数量
        """
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            db.session.bulk_insert_mappings(model_class, batch)
            db.session.commit()
            print(f"Inserted batch {i//batch_size + 1}/{(len(data_list)-1)//batch_size + 1}")
    
    @staticmethod
    def bulk_update(model_class, data_list: List[dict], batch_size: int = 1000):
        """
        批量更新数据
        
        Args:
            model_class: 模型类
            data_list: 数据列表
            batch_size: 每批更新数量
        """
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            db.session.bulk_update_mappings(model_class, batch)
            db.session.commit()
            print(f"Updated batch {i//batch_size + 1}/{(len(data_list)-1)//batch_size + 1}")


class SimpleCache:
    """简单内存缓存"""
    
    def __init__(self, default_ttl: int = 300):
        """
        初始化缓存
        
        Args:
            default_ttl: 默认过期时间（秒）
        """
        self._cache = {}
        self._ttl = default_ttl
    
    def _make_key(self, key: str) -> str:
        """生成缓存键"""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
    
    def get(self, key: str, default=None):
        """获取缓存值"""
        cache_key = self._make_key(key)
        
        if cache_key in self._cache:
            value, expire_time = self._cache[cache_key]
            if time.time() < expire_time:
                return value
            else:
                # 过期删除
                del self._cache[cache_key]
        
        return default
    
    def set(self, key: str, value: Any, ttl: int = None):
        """设置缓存值"""
        cache_key = self._make_key(key)
        expire_time = time.time() + (ttl or self._ttl)
        self._cache[cache_key] = (value, expire_time)
    
    def delete(self, key: str):
        """删除缓存"""
        cache_key = self._make_key(key)
        self._cache.pop(cache_key, None)
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
    
    def cached(self, ttl: int = None):
        """缓存装饰器"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # 尝试获取缓存
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # 执行函数
                result = func(*args, **kwargs)
                
                # 缓存结果
                self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator


# 全局缓存实例
cache = SimpleCache()


def optimize_chunk_query(query, include_document: bool = True):
    """
    优化 Chunk 查询
    
    Args:
        query: Chunk 查询对象
        include_document: 是否预加载 Document
    
    Returns:
        优化后的查询
    """
    if include_document:
        from app.db.models import Document
        query = query.options(joinedload(Chunk.document))
    
    return query


def get_chunks_by_ids_optimized(chunk_ids: List[str], include_document: bool = True):
    """
    优化的批量获取 Chunk 方法
    
    Args:
        chunk_ids: Chunk ID 列表
        include_document: 是否包含 Document 信息
    
    Returns:
        Chunk 列表
    """
    from app.db.models import Chunk, Document
    
    query = Chunk.query.filter(Chunk.chunk_id.in_(chunk_ids))
    
    if include_document:
        query = query.options(joinedload(Chunk.document))
    
    return query.all()


class QueryProfiler:
    """查询性能分析器"""
    
    def __init__(self):
        self.queries = []
    
    def profile(self, func: Callable):
        """性能分析装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            duration = end_time - start_time
            
            query_info = {
                'function': func.__name__,
                'duration': duration,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            
            self.queries.append(query_info)
            
            if duration > 1.0:  # 慢查询警告
                print(f"[Slow Query Warning] {func.__name__} took {duration:.2f}s")
            
            return result
        return wrapper
    
    def get_stats(self):
        """获取统计信息"""
        if not self.queries:
            return {"message": "No queries recorded"}
        
        total_queries = len(self.queries)
        total_time = sum(q['duration'] for q in self.queries)
        avg_time = total_time / total_queries
        max_time = max(q['duration'] for q in self.queries)
        
        return {
            'total_queries': total_queries,
            'total_time': f"{total_time:.2f}s",
            'average_time': f"{avg_time:.2f}s",
            'max_time': f"{max_time:.2f}s",
            'slow_queries': [q for q in self.queries if q['duration'] > 1.0]
        }
    
    def clear(self):
        """清空记录"""
        self.queries.clear()


# 全局性能分析器
profiler = QueryProfiler()
