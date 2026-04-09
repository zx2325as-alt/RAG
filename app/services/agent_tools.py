import json
import os
import shlex
import subprocess
from urllib.parse import urlparse

import requests
from flask import current_app
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import Config
from app.db.models import Chunk, Document


def _json_output(payload):
    return json.dumps(payload, ensure_ascii=False, default=str)


def _truncate_text(text, limit=400):
    if text is None:
        return ""
    text = str(text).strip().replace("\r", " ").replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def _runtime_db_names():
    runtime_context = getattr(current_app, "tool_runtime_context", {}) if current_app else {}
    db_names = runtime_context.get("db_names") or ["default"]
    return [db_name for db_name in db_names if db_name]


class SearchKnowledgeBaseInput(BaseModel):
    query: str = Field(..., description="需要在知识库中检索的完整问题，优先传入改写后的独立问题")
    db_names: list[str] = Field(default_factory=list, description="要检索的知识库列表；为空时自动使用当前对话选中的知识库")
    top_k: int = Field(default=5, ge=1, le=8, description="返回结果条数，建议 3 到 5")


def search_knowledge_base(query: str, db_names: list[str] = None, top_k: int = 5) -> str:
    selected_dbs = db_names or _runtime_db_names()
    kb_service = getattr(getattr(current_app, "qa_service", None), "kb_service", None)
    if kb_service is None:
        from app.services.knowledge_base_service import KnowledgeBaseService
        kb_service = KnowledgeBaseService()

    results = kb_service.search(query, top_k=min(max(top_k, 1), 8), db_names=selected_dbs)
    payload = {
        "query": query,
        "db_names": selected_dbs,
        "result_count": len(results),
        "results": []
    }
    for index, (doc, score) in enumerate(results, 1):
        payload["results"].append({
            "rank": index,
            "doc_id": doc.metadata.get("doc_id"),
            "doc_name": doc.metadata.get("doc_name", "未知文档"),
            "db_name": doc.metadata.get("db_name", "default"),
            "score": round(float(score), 6),
            "content": _truncate_text(doc.page_content, 500)
        })
    return _json_output(payload)


class GetDocumentDetailInput(BaseModel):
    doc_name: str = Field(default="", description="文档名称，已知文档名称时优先使用")
    doc_id: int | None = Field(default=None, description="文档ID，已知文档ID时可直接使用")
    db_name: str = Field(default="", description="文档所在知识库，可为空")
    max_chunks: int = Field(default=3, ge=1, le=8, description="最多返回多少个文档片段")


def get_document_details(doc_name: str = "", doc_id: int | None = None, db_name: str = "", max_chunks: int = 3) -> str:
    query = Document.query
    if doc_id:
        query = query.filter(Document.doc_id == doc_id)
    elif doc_name:
        query = query.filter(Document.doc_name == doc_name)
    else:
        return _json_output({"error": "必须提供 doc_id 或 doc_name"})

    if db_name:
        query = query.filter(Document.db_name == db_name)

    document = query.order_by(Document.upload_time.desc()).first()
    if not document:
        return _json_output({"error": "未找到对应文档"})

    chunks = (
        Chunk.query.filter(Chunk.doc_id == document.doc_id)
        .order_by(Chunk.chunk_index.asc())
        .limit(min(max_chunks, 8))
        .all()
    )
    return _json_output({
        "doc_id": document.doc_id,
        "doc_name": document.doc_name,
        "doc_type": document.doc_type,
        "db_name": document.db_name,
        "status": document.status,
        "upload_time": document.upload_time.isoformat() if document.upload_time else None,
        "chunk_count": len(document.chunks),
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "content": _truncate_text(chunk.content, 700)
            }
            for chunk in chunks
        ]
    })


class QueryMetricsInput(BaseModel):
    service_name: str = Field(default="rag", description="要查询的系统名称，例如 rag、knowledge_base、training、deploy、redis")


def query_metrics(service_name: str = "rag") -> str:
    from app.api.common import active_finetune_processes
    try:
        from app.api.finetune_routes import active_vllm_processes
    except Exception:
        active_vllm_processes = {}

    doc_count = Document.query.count()
    chunk_count = Chunk.query.count()
    service_name_lower = (service_name or "rag").lower()
    payload = {
        "service_name": service_name,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "active_llm": Config.ACTIVE_LLM,
        "current_model": getattr(Config, "OLLAMA_MODEL_NAME", None) if Config.ACTIVE_LLM == "ollama" else (
            getattr(Config, "VLLM_MODEL_NAME", None) if Config.ACTIVE_LLM == "vllm" else getattr(Config, "QWEN_MODEL_NAME", None)
        ),
        "training_tasks": {},
        "deployments": []
    }

    for task_name, process_info in active_finetune_processes.items():
        process = process_info.get("process")
        payload["training_tasks"][task_name] = {
            "running": bool(process and process.poll() is None),
            "model_name": process_info.get("model_name"),
            "start_time": process_info.get("start_time")
        }

    for deployment_name, process in active_vllm_processes.items():
        payload["deployments"].append({
            "model_name": deployment_name,
            "running": process.poll() is None
        })

    if "redis" in service_name_lower:
        qa_service = getattr(current_app, "qa_service", None)
        payload["redis_available"] = bool(qa_service and qa_service.use_redis)
    if "knowledge" in service_name_lower:
        payload["knowledge_bases"] = sorted({doc.db_name for doc in Document.query.all()})
    return _json_output(payload)


class ExecuteShellCommandInput(BaseModel):
    command: str = Field(..., description="只读型系统命令，例如 ping、curl、nslookup、df、free、netstat、ss")


def execute_shell_command(command: str) -> str:
    command = (command or "").strip()
    if not command:
        return _json_output({"error": "命令不能为空"})

    forbidden_tokens = ["rm", "reboot", "shutdown", "mkfs", ">", ">>", "|", ";", "&", "chmod", "chown"]
    lower_command = command.lower()
    if any(token in lower_command for token in forbidden_tokens):
        return _json_output({"error": "命令因安全限制被拒绝"})

    try:
        parts = shlex.split(command, posix=os.name != "nt")
    except ValueError as exc:
        return _json_output({"error": f"命令解析失败: {exc}"})

    if not parts:
        return _json_output({"error": "命令不能为空"})

    allowed_prefixes = {
        "ping", "curl", "nslookup", "traceroute", "tracert", "df", "free",
        "netstat", "ss", "ipconfig", "ifconfig", "route", "uname", "ps"
    }
    if parts[0].lower() not in allowed_prefixes:
        return _json_output({"error": f"仅允许执行只读诊断命令，当前命令 [{parts[0]}] 不在允许列表中"})

    try:
        result = subprocess.run(
            parts,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=12
        )
        return _json_output({
            "command": command,
            "returncode": result.returncode,
            "stdout": _truncate_text(result.stdout, 2000),
            "stderr": _truncate_text(result.stderr, 1200)
        })
    except subprocess.TimeoutExpired:
        return _json_output({"command": command, "error": "命令执行超时"})
    except Exception as exc:
        return _json_output({"command": command, "error": str(exc)})


class QueryApiEndpointInput(BaseModel):
    url: str = Field(..., description="要访问的 HTTP/HTTPS 接口地址")
    method: str = Field(default="GET", description="HTTP 方法，仅支持 GET 或 POST")
    params: dict = Field(default_factory=dict, description="查询参数或 POST JSON 参数")


def query_api_endpoint(url: str, method: str = "GET", params: dict = None) -> str:
    parsed = urlparse(url or "")
    if parsed.scheme not in {"http", "https"}:
        return _json_output({"error": "仅允许访问 http 或 https 接口"})

    params = params or {}
    method = (method or "GET").upper()
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=params, timeout=10)
        else:
            return _json_output({"error": f"不支持的 HTTP 方法: {method}"})

        response.raise_for_status()
        return _json_output({
            "url": url,
            "method": method,
            "status_code": response.status_code,
            "body": _truncate_text(response.text, 2000)
        })
    except Exception as exc:
        return _json_output({"url": url, "method": method, "error": str(exc)})


AVAILABLE_TOOLS = {
    "search_knowledge_base": StructuredTool.from_function(
        func=search_knowledge_base,
        name="search_knowledge_base",
        description="检索当前问答页面所选知识库中的真实资料片段，返回文档名、知识库名、得分和内容摘要。",
        args_schema=SearchKnowledgeBaseInput
    ),
    "get_document_details": StructuredTool.from_function(
        func=get_document_details,
        name="get_document_details",
        description="根据文档名或文档ID读取数据库中的真实文档信息和关键片段，用于追溯证据细节。",
        args_schema=GetDocumentDetailInput
    ),
    "query_metrics": StructuredTool.from_function(
        func=query_metrics,
        name="query_metrics",
        description="查询当前系统的真实状态，包括文档数量、当前模型、训练任务和部署状态。",
        args_schema=QueryMetricsInput
    ),
    "execute_shell_command": StructuredTool.from_function(
        func=execute_shell_command,
        name="execute_shell_command",
        description="执行只读型诊断命令，适用于 ping、curl、df、free、netstat、ss 等运维排查命令。",
        args_schema=ExecuteShellCommandInput
    ),
    "query_api_endpoint": StructuredTool.from_function(
        func=query_api_endpoint,
        name="query_api_endpoint",
        description="访问指定的 HTTP 或 HTTPS 接口并返回结构化响应结果。",
        args_schema=QueryApiEndpointInput
    )
}


def get_tools_by_names(tool_names):
    unique_names = []
    for tool_name in tool_names:
        if tool_name in AVAILABLE_TOOLS and tool_name not in unique_names:
            unique_names.append(tool_name)
    return [AVAILABLE_TOOLS[name] for name in unique_names]
