import os
import sys
import warnings
import logging

# 将项目根目录添加到 sys.path，以便能够正确导入 app 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.config import Config

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6008, debug=Config.DEBUG)
