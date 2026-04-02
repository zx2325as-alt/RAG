from flask import Flask
from app.config import Config
from app.db import db
import logging
import os
from logging.handlers import RotatingFileHandler

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 在系统启动时自动检查并创建所有必需的运行时目录，避免在新环境中因目录缺失而报错
    required_directories = [
        'logs',
        app.config.get('UPLOAD_FOLDER', 'uploads'),
        'finetuned_models',
        'finetuned_models/base_models',
        'finetuned_models/runs',
        'finetune_configs'
    ]
    
    # 获取项目根目录 (假设 app 目录和根目录同级，这里我们计算出根目录路径)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for directory in required_directories:
        dir_path = os.path.join(root_dir, directory)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {dir_path}: {e}")

    # 设置详细的日志格式
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/rag_system.log', maxBytes=10240000, backupCount=10)
    # 增加线程名、进程ID等详细信息
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s [%(pathname)s:%(lineno)d] - Thread:%(threadName)s: %(message)s'
    ))
    file_handler.setLevel(logging.DEBUG) # 将日志级别降低到 DEBUG 以记录更多细节
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.DEBUG)
    
    # 将 werkzeug 的日志也重定向到我们的文件中
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.DEBUG)
    log.addHandler(file_handler)
    
    # 避免在 Flask debug 重载器子进程中重复输出启动日志
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        app.logger.info('='*50)
        app.logger.info('RAG System startup - Detailed Logging Enabled')
        app.logger.info('='*50)

    db.init_app(app)

    # Register Blueprints
    with app.app_context():
        from app.db import models
        db.create_all()
        from app.api.routes import api_bp
        from app.api.eval_routes import eval_bp
        app.register_blueprint(api_bp, url_prefix='/')
        app.register_blueprint(eval_bp, url_prefix='/')

    return app
