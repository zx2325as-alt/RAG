from flask import Flask
from app.config import Config
from app.db import db
import logging
import os
from logging.handlers import RotatingFileHandler

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

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
        from app.api.routes import api_bp
        app.register_blueprint(api_bp, url_prefix='/') # Mount at root for simplicity or keep /api for API and another for UI

    return app
