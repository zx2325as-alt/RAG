import os
import sys
import warnings
import logging

# # 忽略第三方库 (如 jieba) 带来的 pkg_resources 弃用警告
# warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# # 屏蔽 PaddlePaddle 烦人的 Connectivity check 提示
# os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = '1'
# os.environ['PADDLE_DISABLE_HPC_LOG'] = '1'
# logging.getLogger("paddle").setLevel(logging.ERROR)

# 将项目根目录添加到 sys.path，以便能够正确导入 app 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.config import Config

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=Config.DEBUG)
