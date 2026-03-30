import re
import json

def clean_text(text):
    """
    Clean text by removing extra whitespace, noise, etc.
    """
    # Remove null characters
    text = text.replace('\x00', '')
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_text_type_and_adjust_size(text, default_size=300):
    """动态分块大小：根据内容密度、是否包含代码块或JSON等调整大小"""
    if "```" in text or "{" in text and "}" in text:
        # 代码块或 JSON 结构，适当扩大块大小以保证完整性
        return 800, 100
    if len(re.findall(r'[a-zA-Z0-9_]', text)) / (len(text) + 1) > 0.6:
        # 英文/日志较多
        return 600, 100
    return default_size, 50

def chunk_text(text, chunk_size=300, overlap=50, is_markdown=False, metadata=None):
    """
    Split text into chunks with overlap using LangChain's RecursiveCharacterTextSplitter.
    优化：动态分块大小、增强元数据注入、基于句子边界的语义切分
    """
    if not text:
        return []
        
    from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    
    # 动态分块大小与重叠策略优化（基于句子边界的重叠）
    dynamic_size, dynamic_overlap = detect_text_type_and_adjust_size(text, chunk_size)
    
    # 增强元数据前缀
    meta_prefix = ""
    if metadata:
        meta_prefix = "【文档元数据】" + " | ".join([f"{k}:{v}" for k, v in metadata.items()]) + "\n"

    # 优先在句子级别切分（语义切分的基础），确保句子不被截断
    separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", ";", " ", ""]

    if is_markdown:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=dynamic_size,
            chunk_overlap=dynamic_overlap,
            separators=separators
        )
        chunks = []
        for split in md_header_splits:
            # 组合 header metadata 与外部增强 metadata
            header_context = " ".join([f"{k}: {v}" for k, v in split.metadata.items()])
            sub_chunks = text_splitter.split_text(split.page_content)
            for sub in sub_chunks:
                final_chunk = f"{meta_prefix}{header_context}\n{sub}".strip()
                chunks.append(final_chunk)
        return chunks
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=dynamic_size,
            chunk_overlap=dynamic_overlap,
            separators=separators
        )
        
        raw_chunks = text_splitter.split_text(text)
        if meta_prefix:
            return [f"{meta_prefix}{c}".strip() for c in raw_chunks]
        return raw_chunks
