import os

file_path = r'E:\python\condaEnv\langchain\lib\site-packages\gradio_client\utils.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'def get_type(schema: dict):\n    if "const" in schema:',
    'def get_type(schema: dict):\n    if isinstance(schema, bool):\n        return {}\n    if "const" in schema:'
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Patch applied')
