import fitz  # PyMuPDF
import os
import pdfplumber
import logging

# 忽略 pdfminer 的 FontBBox 警告
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# 忽略 onnxruntime 的执行提供者警告
import os
os.environ["ONNXRUNTIME_WARNINGS"] = "0"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

# 尝试导入 chardet 用于编码自动检测
try:
    import chardet
except ImportError:
    chardet = None
    logging.warning("chardet not installed, falling back to utf-8 for text file encoding detection")

# 初始化 RapidOCR（全局单例，避免重复加载）
# 需要安装：pip install rapidocr-onnxruntime
try:
    from rapidocr_onnxruntime import RapidOCR
    import numpy as np
    import torch
    import sys
    import os
    
    # 彻底屏蔽 OrtInferSession 在底层硬编码的 logging
    import logging
    logging.getLogger('OrtInferSession').setLevel(logging.FATAL)
    
    # 检测是否有 GPU，如果可用，通过 onnxruntime 开启 GPU 推理支持
    # 这需要环境中安装了 onnxruntime-gpu
    use_gpu = torch.cuda.is_available()
    
    # 暂时重定向 stderr 以彻底屏蔽 onnxruntime 底层硬编码的 WARNING 和 INFO 刷屏
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    if use_gpu:
        try:
            # 尝试初始化带 CUDA Execution Provider 的 OCR
            ocr = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
            sys.stderr = _stderr
            print("RapidOCR initialized with GPU (CUDA) support.")
        except Exception as gpu_e:
            sys.stderr = _stderr
            print(f"Failed to initialize RapidOCR with GPU: {gpu_e}. Falling back to CPU.")
            ocr = RapidOCR()
    else:
        ocr = RapidOCR()
        sys.stderr = _stderr
        
except Exception as e:
    ocr = None
    print(f"Warning: OCR initialization failed ({e}). OCR fallback will be disabled.")

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
                    # 表格有效性验证：至少2行2列
                    if len(table) < 2:
                        logging.debug(f"Skipping table with only {len(table)} row(s)")
                        continue
                    # 检查是否至少有一行有2个或以上单元格
                    max_cols = max(len(row) for row in table) if table else 0
                    if max_cols < 2:
                        logging.debug(f"Skipping table with only {max_cols} column(s)")
                        continue
                    # 检查表头是否全为空
                    header_row = table[0] if table else []
                    header_values = [str(cell).strip() if cell else "" for cell in header_row]
                    if all(not cell for cell in header_values):
                        logging.debug("Skipping table with empty header row")
                        continue
                    # 将表格转换为 Markdown 格式
                    for i, row in enumerate(table):
                        # 过滤掉 None，转为字符串
                        clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                        # 记录 None 单元格用于调试
                        none_cells = [j for j, cell in enumerate(row) if cell is None]
                        if none_cells:
                            logging.debug(f"Row {i} has None cells at positions: {none_cells}")
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
        
        # 提取页面中的所有图片并进行 OCR 识别，确保图片内的公式、文字不丢失
        image_list = []
        if ocr is not None:
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # 确保图片色彩空间受支持 (如果是 CMYK 等，转换为 RGB)
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    
                    if pix.n == 4:
                        import cv2
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                    elif pix.n == 1:
                        import cv2
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                        
                    result, _ = ocr(img_array)
                    if result:
                        text += "\n[图片补充内容]: "
                        for line in result:
                            text += line[1] + " "
                        text += "\n"
                except Exception as e:
                    # 单张图片OCR失败只记录warning，不影响其他图片和整页处理
                    logging.warning(f"OCR failed on image {img_index} in page {page_num}: {e}")
                    
        # 兼容处理全扫描件：如果一页文本极少，且没有提取到明显的内嵌图片，则将整页渲染后进行 OCR
        if len(text.strip()) < 50 and ocr is not None and not image_list:
            try:
                # 渲染页面为图片
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 提高分辨率
                # 将 pixmap 转换为 numpy 数组
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                # 如果是 RGBA，转为 RGB
                if pix.n == 4:
                    import cv2
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif pix.n == 1: # 灰度图
                    import cv2
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    
                result, _ = ocr(img)
                if result:
                    for line in result:
                        text += line[1] + "\n"
            except Exception as e:
                print(f"OCR failed on page {page_num}: {e}")
            
        full_text += text + "\n"
        
    return full_text

def extract_text_from_image(file_path):
    """
    Extract text directly from images using OCR.
    """
    if ocr is None:
        return "Error: rapidocr-onnxruntime not installed. Cannot process image."
    
    full_text = ""
    result, _ = ocr(file_path)
    if result:
        for line in result:
            full_text += line[1] + "\n"
    return full_text

def extract_text_from_docx(file_path):
    """
    Extract text from Word (.docx) files, preserving tables as Markdown.
    """
    if DocxDocument is None:
        return "Error: python-docx not installed."
        
    doc = DocxDocument(file_path)
    full_text = []
    
    # Process elements in order (paragraphs and tables)
    # doc.element.body contains all elements in order
    for element in doc.element.body:
        if element.tag.endswith('p'):
            # This is a paragraph
            from docx.text.paragraph import Paragraph
            para = Paragraph(element, doc)
            if para.text.strip():
                full_text.append(para.text)
        elif element.tag.endswith('tbl'):
            # This is a table
            from docx.table import Table
            table = Table(element, doc)
            
            for i, row in enumerate(table.rows):
                clean_row = [cell.text.replace('\n', ' ').strip() for cell in row.cells]
                full_text.append("| " + " | ".join(clean_row) + " |")
                if i == 0:
                    full_text.append("|" + "|".join(["---"] * len(clean_row)) + "|")
            full_text.append("") # Empty line after table

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
        # 使用 chardet 自动检测编码，如果不可用则降级为 utf-8
        if chardet:
            with open(file_path, 'rb') as f:
                raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected.get('encoding', 'utf-8') or 'utf-8'
            logging.debug(f"Detected encoding for {file_path}: {encoding}")
        else:
            encoding = 'utf-8'
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            return f.read()
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        return extract_text_from_image(file_path)
    else:
        return ""
