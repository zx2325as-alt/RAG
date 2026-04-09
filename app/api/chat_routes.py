"""
纯净聊天相关路由：不带知识库检索的对话功能
"""
from flask import request, jsonify, Response, current_app, render_template
from app.api.common import api_bp, get_qa_service
import json

@api_bp.route('/chat')
def chat_page():
    return render_template('chat.html')

@api_bp.route('/pure_chat', methods=['POST'])
def pure_chat():
    data = request.json
    question = data.get('question')
    user_id = request.remote_addr
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        from flask import stream_with_context
        qa_service = get_qa_service()

        def safe_stream():
            try:
                # 获取历史记录
                history = qa_service.get_session_history(f"pure_chat_{user_id}")
                messages = history.messages
                
                # 构建简单的 prompt
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                
                system_msg = SystemMessage(content="你是一个有用的AI助手。请直接回答用户的问题。")
                
                # 准备发送给LLM的消息列表
                chat_messages = [system_msg]
                chat_messages.extend(messages)
                chat_messages.append(HumanMessage(content=question))
                
                # 流式生成
                full_response = ""
                for chunk in qa_service.llm.stream(chat_messages):
                    content = chunk.content
                    if content:
                        full_response += content
                        yield json.dumps({"type": "chunk", "content": content}) + "\n"
                
                # 保存到历史记录
                history.add_user_message(question)
                history.add_ai_message(full_response)
                
            except Exception as inner_e:
                import traceback
                current_app.logger.error(f"Stream error: {traceback.format_exc()}")
                yield json.dumps({"type": "chunk", "content": f"\n\n> [!ERROR] 生成时发生错误: {str(inner_e)}\n\n"}) + "\n"

        return Response(stream_with_context(safe_stream()), mimetype='application/x-ndjson')
    except Exception as e:
        import traceback
        current_app.logger.error(f"[/pure_chat] Error occurred: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/pure_chat/clear', methods=['POST'])
def clear_pure_chat():
    user_id = request.remote_addr
    try:
        qa_service = get_qa_service()
        history = qa_service.get_session_history(f"pure_chat_{user_id}")
        history.clear()
        return jsonify({'status': 'ok', 'message': 'Chat history cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
