"""
统一异常处理模块
定义项目中使用的所有自定义异常类
"""


class RAGException(Exception):
    """RAG 系统基础异常类"""
    
    def __init__(self, message: str, error_code: str = "RAG_001", status_code: int = 500, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """转换为字典格式，用于 API 响应"""
        return {
            'error': self.error_code,
            'message': self.message,
            'details': self.details
        }


# ==================== 文档相关异常 ====================

class DocumentNotFoundError(RAGException):
    """文档不存在"""
    def __init__(self, doc_id: str = None):
        super().__init__(
            message=f"Document not found" + (f": {doc_id}" if doc_id else ""),
            error_code="DOC_001",
            status_code=404
        )


class DocumentProcessingError(RAGException):
    """文档处理错误"""
    def __init__(self, message: str = "Document processing failed"):
        super().__init__(
            message=message,
            error_code="DOC_002",
            status_code=500
        )


class UnsupportedFileTypeError(RAGException):
    """不支持的文件类型"""
    def __init__(self, file_type: str = None):
        super().__init__(
            message=f"Unsupported file type" + (f": {file_type}" if file_type else ""),
            error_code="DOC_003",
            status_code=400
        )


# ==================== 检索相关异常 ====================

class IndexNotFoundError(RAGException):
    """索引不存在"""
    def __init__(self, db_name: str = None):
        super().__init__(
            message=f"Index not found" + (f" for database: {db_name}" if db_name else ""),
            error_code="IDX_001",
            status_code=404
        )


class SearchError(RAGException):
    """检索错误"""
    def __init__(self, message: str = "Search operation failed"):
        super().__init__(
            message=message,
            error_code="SRCH_001",
            status_code=500
        )


# ==================== 模型相关异常 ====================

class ModelNotFoundError(RAGException):
    """模型不存在"""
    def __init__(self, model_name: str = None):
        super().__init__(
            message=f"Model not found" + (f": {model_name}" if model_name else ""),
            error_code="MDL_001",
            status_code=404
        )


class ModelLoadError(RAGException):
    """模型加载错误"""
    def __init__(self, message: str = "Failed to load model"):
        super().__init__(
            message=message,
            error_code="MDL_002",
            status_code=500
        )


class LLMError(RAGException):
    """大语言模型调用错误"""
    def __init__(self, message: str = "LLM operation failed"):
        super().__init__(
            message=message,
            error_code="LLM_001",
            status_code=500
        )


# ==================== 微调相关异常 ====================

class FinetuneError(RAGException):
    """微调训练错误"""
    def __init__(self, message: str = "Finetune operation failed"):
        super().__init__(
            message=message,
            error_code="FT_001",
            status_code=500
        )


class DatasetNotFoundError(RAGException):
    """数据集不存在"""
    def __init__(self, dataset_name: str = None):
        super().__init__(
            message=f"Dataset not found" + (f": {dataset_name}" if dataset_name else ""),
            error_code="DATA_001",
            status_code=404
        )


# ==================== 配置相关异常 ====================

class ConfigError(RAGException):
    """配置错误"""
    def __init__(self, message: str = "Configuration error"):
        super().__init__(
            message=message,
            error_code="CFG_001",
            status_code=500
        )


# ==================== 权限相关异常 ====================

class PermissionDeniedError(RAGException):
    """权限不足"""
    def __init__(self, message: str = "Permission denied"):
        super().__init__(
            message=message,
            error_code="AUTH_001",
            status_code=403
        )


class AuthenticationError(RAGException):
    """认证失败"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTH_002",
            status_code=401
        )


# ==================== 验证相关异常 ====================

class ValidationError(RAGException):
    """数据验证错误"""
    def __init__(self, message: str = "Validation failed", details: dict = None):
        super().__init__(
            message=message,
            error_code="VAL_001",
            status_code=400,
            details=details
        )
