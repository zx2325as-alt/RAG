import os

file_path = r'E:\python\condaEnv\langchain\lib\site-packages\gradio_client\utils.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换 _json_schema_to_python_type 的开头
target_str = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if schema == {}:'

new_str = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if isinstance(schema, bool):\n        return "Any"\n    if schema == {}:'

content = content.replace(target_str, new_str)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Patch 2 applied')
