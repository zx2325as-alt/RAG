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
                app.logger.warning(f"Warning: Could not create directory {dir_path}: {e}")

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
    
    app.logger.info('='*50)
    app.logger.info('RAG System startup - Detailed Logging Enabled')
    app.logger.info('='*50)

    db.init_app(app)

    # Register Blueprints
    with app.app_context():
        from app.db import models
        db.create_all()
        
        # 统一异常处理
        from flask import jsonify
        from werkzeug.exceptions import HTTPException

        @app.errorhandler(Exception)
        def handle_exception(e):
            # 处理 HTTP 异常 (如 404, 405 等)
            if isinstance(e, HTTPException):
                response = {
                    "code": e.code,
                    "message": e.description,
                    "status": "error"
                }
                return jsonify(response), e.code

            # 处理业务逻辑中的未捕获异常
            app.logger.error(f"Unhandled Exception: {str(e)}", exc_info=True)
            response = {
                "code": 500,
                "message": "Internal Server Error",
                "status": "error"
            }
            # 如果是开发模式，可以返回更详细的错误信息
            if app.debug:
                response["detail"] = str(e)
            
            return jsonify(response), 500

        from app.api.routes import api_bp
        app.register_blueprint(api_bp, url_prefix='/') # Mount at root for simplicity or keep /api for API and another for UI

    return app
