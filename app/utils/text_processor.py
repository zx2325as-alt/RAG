import re

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

def chunk_text(text, chunk_size=300, overlap=50, is_markdown=False):
    """
    Split text into chunks with overlap using LangChain's RecursiveCharacterTextSplitter.
    If is_markdown is True, use MarkdownHeaderTextSplitter first for semantic chunking.
    块大小300字符，重叠50字符
    """
    if not text:
        return []
        
    from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    
    if is_markdown:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(text)
        
        # 进一步细分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        chunks = []
        for split in md_header_splits:
            # 组合 header metadata
            header_context = " ".join([f"{k}: {v}" for k, v in split.metadata.items()])
            sub_chunks = text_splitter.split_text(split.page_content)
            for sub in sub_chunks:
                chunks.append(f"{header_context}\n{sub}".strip())
        return chunks
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        return chunks
