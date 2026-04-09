"""
API 路由入口文件 - 汇总所有子模块路由

模块划分：
- common.py: 公共工具函数、Blueprint 定义、共享变量
- system_routes.py: 首页、系统状态、TensorBoard 管理
- document_routes.py: 文档上传、删除、查询
- llm_routes.py: LLM 模型设置、查询、反馈
- finetune_routes.py: 微调训练、评估、数据集管理
"""
from app.api.common import api_bp, register_routes

# 注册所有路由模块
register_routes()
