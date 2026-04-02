#!/usr/bin/env python3
"""
Neo4j 服务启动和检查工具
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# Neo4j 安装路径
NEO4J_HOME = Path(r"E:\python\conda\RAG\model\neo4j-community-5.18.1")

# Java 路径（Neo4j 5.x 需要 Java 17+）
# 如果系统已安装 Java，可以设为 None 让 Neo4j 自动检测
JAVA_HOME = None  # 例如: Path(r"C:\Program Files\Java\jdk-17")

def check_neo4j_running():
    """检查 Neo4j 是否已在运行"""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "11111111"))
        with driver.session() as session:
            result = session.run("RETURN 1 as num")
            result.single()
        driver.close()
        return True
    except Exception as e:
        print(f"Neo4j 连接检查失败: {e}")
        return False

def start_neo4j():
    """启动 Neo4j 服务"""
    if sys.platform == 'win32':
        neo4j_script = NEO4J_HOME / "bin" / "neo4j.bat"
    else:
        neo4j_script = NEO4J_HOME / "bin" / "neo4j"
    
    if not neo4j_script.exists():
        print(f"错误: 找不到 Neo4j 启动脚本: {neo4j_script}")
        return False
    
    print(f"正在启动 Neo4j...")
    print(f"脚本路径: {neo4j_script}")
    
    try:
        # 使用 console 模式启动（前台运行，方便查看日志）
        result = subprocess.run(
            [str(neo4j_script), "console"],
            cwd=str(NEO4J_HOME),
            capture_output=True,
            text=True,
            timeout=30
        )
        print(result.stdout)
        if result.stderr:
            print(f"错误输出: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("启动超时，但 Neo4j 可能已在后台启动")
        return True
    except Exception as e:
        print(f"启动失败: {e}")
        return False

def init_neo4j_password():
    """初始化 Neo4j 密码"""
    admin_script = NEO4J_HOME / "bin" / "neo4j-admin.bat"
    
    if not admin_script.exists():
        print(f"错误: 找不到 neo4j-admin 脚本")
        return False
    
    print("正在设置 Neo4j 密码...")
    try:
        # Neo4j 5.x 使用 set-initial-password
        result = subprocess.run(
            [str(admin_script), "dbms", "set-initial-password", "11111111"],
            cwd=str(NEO4J_HOME),
            capture_output=True,
            text=True,
            timeout=30
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"设置密码失败: {result.stderr}")
            # 如果失败，可能是密码已设置，忽略错误
            return True
        return True
    except Exception as e:
        print(f"设置密码出错: {e}")
        return False

def check_java():
    """检查 Java 环境"""
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return True, result.stderr or result.stdout
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("Neo4j 服务管理工具")
    print("=" * 60)
    
    # 检查 Neo4j 目录
    if not NEO4J_HOME.exists():
        print(f"错误: Neo4j 目录不存在: {NEO4J_HOME}")
        sys.exit(1)
    
    print(f"Neo4j 安装路径: {NEO4J_HOME}")
    
    # 检查 Java 环境
    print("\n0. 检查 Java 环境...")
    has_java, java_info = check_java()
    if has_java:
        print(f"✓ Java 已安装")
        print(f"  {java_info.split(chr(10))[0] if java_info else ''}")
    else:
        print("✗ Java 未安装或未配置环境变量")
        print()
        print("=" * 60)
        print("【重要】Neo4j 5.x 需要 Java 17 或更高版本")
        print("=" * 60)
        print()
        print("请按以下步骤安装 Java：")
        print()
        print("方法1: 使用 conda 安装（推荐）")
        print("  conda install -c conda-forge openjdk=17")
        print()
        print("方法2: 手动下载安装")
        print("  1. 访问 https://adoptium.net/")
        print("  2. 下载 OpenJDK 17 (LTS)")
        print("  3. 安装并配置环境变量 JAVA_HOME")
        print()
        print("安装完成后，重新运行此脚本")
        return
    
    # 检查是否已在运行
    print("\n1. 检查 Neo4j 是否已在运行...")
    if check_neo4j_running():
        print("✓ Neo4j 已在运行，可以直接使用！")
        print()
        print("图谱功能已就绪。要构建知识图谱，需要：")
        print("  1. 在文档上传时启用'构建知识图谱'选项")
        print("  2. 或在已有知识库上运行图谱构建任务")
        return
    else:
        print("✗ Neo4j 未运行")
    
    # 尝试启动
    print("\n2. 尝试启动 Neo4j...")
    print("注意: 如果是首次启动，可能需要设置密码")
    print("      默认用户名: neo4j")
    print("      默认密码: 11111111")
    print()
    
    # 检查数据目录
    data_dir = NEO4J_HOME / "data" / "databases"
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("检测到首次启动，需要初始化...")
        init_neo4j_password()
    
    print("\n" + "=" * 60)
    print("启动命令:")
    print("=" * 60)
    print(f"  cd {NEO4J_HOME}")
    print(f"  bin\\neo4j.bat console")
    print()
    print("或者使用后台模式:")
    print(f"  bin\\neo4j.bat start")
    print()
    print("=" * 60)
    print("启动后，可以通过以下地址访问:")
    print("=" * 60)
    print("  - 浏览器界面: http://localhost:7474")
    print("  - Bolt 协议: bolt://localhost:7687")
    print()
    print("默认登录信息:")
    print("  用户名: neo4j")
    print("  密码: 11111111")
    print()
    print("=" * 60)
    print("【下一步】构建知识图谱")
    print("=" * 60)
    print("启动 Neo4j 后，需要构建知识图谱才能使用图谱检索功能：")
    print()
    print("方法1: 上传新文档时启用图谱构建")
    print("  - 在文档上传页面勾选'构建知识图谱'选项")
    print()
    print("方法2: 为已有文档构建图谱")
    print("  - 在知识库管理页面选择'重建图谱'")

if __name__ == "__main__":
    main()
