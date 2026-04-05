"""
系统相关路由：首页、状态、TensorBoard
"""
from flask import jsonify, render_template, current_app
from app.db.models import Document, Chunk
from app.config import Config
from app.api.common import api_bp, tensorboard_process
import os
import sys
import socket
import subprocess


@api_bp.route('/')
def index():
    return render_template('index.html')


@api_bp.route('/finetune')
def finetune():
    return render_template('finetune.html',
                           tensorboard_port=Config.TENSORBOARD_PORT,
                           tensorboard_public_url=Config.TENSORBOARD_PUBLIC_URL)


@api_bp.route('/api/start_tensorboard', methods=['POST'])
def start_tensorboard():
    global tensorboard_process

    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            check_host = '127.0.0.1' if Config.TENSORBOARD_HOST == '0.0.0.0' else Config.TENSORBOARD_HOST
            return s.connect_ex((check_host, port)) == 0

    if is_port_in_use(Config.TENSORBOARD_PORT):
        return jsonify({'status': 'already_running'})

    try:
        log_dir = os.path.join(current_app.root_path, '..', 'finetuned_models', 'runs')
        os.makedirs(log_dir, exist_ok=True)

        tb_cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", log_dir,
            "--host", Config.TENSORBOARD_HOST,
            "--port", str(Config.TENSORBOARD_PORT),
            "--reload_interval", "1"
        ]
        if Config.TENSORBOARD_HOST == '0.0.0.0':
            tb_cmd.append("--bind_all")

        # 增加 TensorBoard 启动超时和重试机制，或者分离进程组
        kwargs = {}
        if os.name != 'nt':
            kwargs['preexec_fn'] = os.setpgrp
            
        tensorboard_process = subprocess.Popen(
            tb_cmd,
            stdout=open(os.path.join(current_app.root_path, '..', 'logs', 'tensorboard_stdout.log'), 'w'),
            stderr=subprocess.STDOUT,
            **kwargs
        )
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/check_tensorboard', methods=['GET'])
def check_tensorboard():
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            check_host = '127.0.0.1' if Config.TENSORBOARD_HOST == '0.0.0.0' else Config.TENSORBOARD_HOST
            return s.connect_ex((check_host, port)) == 0

    return jsonify({'running': is_port_in_use(Config.TENSORBOARD_PORT)})


@api_bp.route('/status', methods=['GET'])
def status():
    doc_count = Document.query.count()
    chunk_count = Chunk.query.count()
    return jsonify({
        'status': 'ok',
        'message': 'RAG System is running',
        'doc_count': doc_count,
        'chunk_count': chunk_count
    })


@api_bp.route('/eval_report')
def eval_report():
    return render_template('eval_report.html')


@api_bp.route('/eval_system')
def eval_system():
    return render_template('eval_system.html')
