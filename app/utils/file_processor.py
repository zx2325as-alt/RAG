import fitz  # PyMuPDF
import os
import pdfplumber

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

# 初始化 PaddleOCR（全局单例，避免重复加载）
# 需要安装：pip install paddlepaddle paddleocr
try:
    # 禁用 Paddle 联网检查
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    from paddleocr import PaddleOCR
    import numpy as np
    import logging
    
    # 禁用 PaddleOCR 默认的 INFO 级别日志输出（包括那些模型下载、缓存提示）
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    import logging as sys_logging
    # PaddleOCR 内部使用了原生的 logging，需要将总日志调低
    sys_logging.getLogger('ppocr').setLevel(sys_logging.WARNING)
    # 当前版本的 PaddleOCR 不支持 show_log 参数，通过上方 sys_logging 设置来控制日志
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
except ImportError:
    ocr = None
    print("Warning: PaddleOCR not installed. OCR fallback will be disabled.")

def extract_text_from_pdf(file_path):
    """
    Extract text from PDF. Use PyMuPDF for text layers, fallback to OCR for images, and pdfplumber for tables.
    """
    full_text = ""
    
    # 1. 优先使用 pdfplumber 提取表格并将其转为 Markdown 格式
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    # 将表格转换为 Markdown 格式
                    for i, row in enumerate(table):
                        # 过滤掉 None，转为字符串
                        clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                        full_text += "| " + " | ".join(clean_row) + " |\n"
                        # 表头分隔线
                        if i == 0:
                            full_text += "|" + "|".join(["---"] * len(clean_row)) + "|\n"
                    full_text += "\n"
    except Exception as e:
        print(f"pdfplumber table extraction failed: {e}")

    # 2. 使用 PyMuPDF 提取普通文本或进行 OCR
    doc = fitz.open(file_path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        
        # 如果提取出的文本过少，可能该页是扫描件或图片，则启用 OCR
        if len(text.strip()) < 50 and ocr is not None:
            # 渲染页面为图片
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 提高分辨率
            # 将 pixmap 转换为 numpy 数组供 PaddleOCR 使用
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            # 如果是 RGBA，转为 RGB
            if pix.n == 4:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1: # 灰度图
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
            result = ocr.ocr(img)
            if result and result[0]:
                for line in result[0]:
                    text += line[1][0] + "\n"
            
        full_text += text + "\n"
        
    return full_text

def extract_text_from_image(file_path):
    """
    Extract text directly from images using OCR.
    """
    if ocr is None:
        return "Error: PaddleOCR not installed. Cannot process image."
    
    full_text = ""
    result = ocr.ocr(file_path)
    if result and result[0]:
        for line in result[0]:
            full_text += line[1][0] + "\n"
    return full_text

def extract_text_from_docx(file_path):
    """
    Extract text from Word (.docx) files.
    """
    if DocxDocument is None:
        return "Error: python-docx not installed."
        
    doc = DocxDocument(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def process_file(file_path):
    """
    Process file based on extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.docx']:
        return extract_text_from_docx(file_path)
    elif ext in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        return extract_text_from_image(file_path)
    else:
        return ""
