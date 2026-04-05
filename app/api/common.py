"""
公共工具函数和共享变量
"""
from flask import Blueprint, current_app
from app.services.document_service import DocumentService
from app.services.qa_service import QAService
import os
import sys
import shutil

# 创建 Blueprint
api_bp = Blueprint('api', __name__)

# 存储 webui 和 tensorboard 进程引用
tensorboard_process = None

# 全局字典，用于存储当前正在运行的微调进程，以便可以强杀
active_finetune_processes = {}


def get_llamafactory_cli_path():
    """获取 llamafactory-cli 的路径"""
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


def get_document_service():
    """懒加载 DocumentService"""
    if not hasattr(current_app, 'document_service'):
        current_app.document_service = DocumentService()
    return current_app.document_service


def get_qa_service():
    """懒加载 QAService"""
    if not hasattr(current_app, 'qa_service'):
        current_app.qa_service = QAService()
    return current_app.qa_service


def get_kb_service():
    """统一使用 qa_service 内部的 kb_service"""
    return get_qa_service().kb_service


# 导入所有子模块的路由（必须在 Blueprint 创建之后）
def register_routes():
    """注册所有路由模块"""
    from app.api import system_routes
    from app.api import document_routes
    from app.api import llm_routes
    from app.api import finetune_routes
    from app.api import chat_routes
