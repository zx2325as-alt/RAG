from langchain_core.tools import tool
import subprocess
import requests

@tool
def execute_shell_command(command: str) -> str:
    """
    执行受限的系统命令，用于查询主机网络状态或系统信息，例如 'ping www.baidu.com', 'df -h', 'free -m', 'top' 等。
    当你识别到用户输入的是一条系统命令（特别是以 ping, df, free 等开头的网络或运维命令）时，必须调用此工具来获取真实的执行结果。
    注意：出于安全原因，禁止执行任何修改系统状态的命令（如 rm, reboot）。
    """
    # 简单的安全过滤
    forbidden_words = ['rm ', 'reboot', 'shutdown', 'mkfs', '>']
    if any(word in command for word in forbidden_words):
        return "Command rejected for security reasons."
    
    try:
        # 使用 subprocess 执行命令并捕获输出
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Command execution timed out."
    except subprocess.CalledProcessError as e:
        return f"Command execution failed: {e.stderr}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

@tool
def query_api_endpoint(url: str, method: str = "GET", params: dict = None) -> str:
    """
    调用外部 REST API 接口获取数据。例如调用 Zabbix API, Prometheus API 或 K8s API 等。
    提供 url, HTTP method (默认 GET), 和可选的参数字典。
    """
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, json=params, timeout=10)
        else:
            return f"Unsupported HTTP method: {method}"
            
        response.raise_for_status()
        return response.text[:2000] # 截断以避免超出大模型上下文
    except Exception as e:
        return f"API request failed: {str(e)}"

@tool
def query_metrics(service_name: str) -> str:
    """
    专门用于查询指定服务（如 Nginx, MySQL, Redis）的健康指标或运行状态。
    建议在决定是否需要重启或执行 shell 命令之前，先调用此工具获取指标进行判断。
    """
    # 模拟指标查询结果
    metrics_mock = {
        "nginx": "Status: Active, CPU: 85%, Memory: 40%, Connections: 1500 (Warning: High CPU load)",
        "mysql": "Status: Active, CPU: 20%, Memory: 60%, QPS: 350",
        "redis": "Status: Down, Error: Connection refused"
    }
    service_name_lower = service_name.lower()
    for key, val in metrics_mock.items():
        if key in service_name_lower:
            return f"Metrics for {service_name}: {val}"
    return f"Metrics for {service_name}: Status Unknown."

# 工具注册表，方便动态加载和扩展
AVAILABLE_TOOLS = {
    "execute_shell_command": execute_shell_command,
    "query_api_endpoint": query_api_endpoint,
    "query_metrics": query_metrics
}

def get_tools_by_names(tool_names):
    """根据配置名称获取对应的工具函数列表"""
    return [AVAILABLE_TOOLS[name] for name in tool_names if name in AVAILABLE_TOOLS]
