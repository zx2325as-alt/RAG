import os
from .local import LocalConfig
from .production import ProductionConfig

# 一键切换配置：通过环境变量 APP_ENV 控制 (local 或 production)
env = os.getenv('APP_ENV', 'local').lower()
config_map = {
    'local': LocalConfig,
    'production': ProductionConfig
}

Config = config_map.get(env, LocalConfig)