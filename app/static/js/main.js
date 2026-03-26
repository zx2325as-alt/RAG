// 全局变量用于存储待上传的文件列表，避免页面切换或多次选择丢失
let pendingFiles = [];

// 配置 marked.js 支持 markdown
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,
        gfm: true
    });
}

// 将 File 对象列表渲染到前端
function renderPendingFiles() {
    var listContainer = $('#selectedFilesList');
    listContainer.empty();
    
    if (pendingFiles.length > 0) {
        pendingFiles.forEach((file, index) => {
            var fileItem = $(`
                <div class="d-flex justify-content-between align-items-center mb-1 p-1 border rounded bg-white">
                    <span class="text-truncate" style="max-width: 80%;" title="${file.name}">📄 ${file.name}</span>
                    <button type="button" class="btn btn-sm btn-outline-danger py-0 px-1 remove-file-btn" data-index="${index}" style="font-size: 0.75rem;">×</button>
                </div>
            `);
            listContainer.append(fileItem);
        });

        // 绑定删除按钮事件
        $('.remove-file-btn').click(function() {
            var index = $(this).data('index');
            pendingFiles.splice(index, 1); // 从数组中移除
            renderPendingFiles(); // 重新渲染列表
        });
    } else {
        listContainer.html('<div class="text-center text-muted">暂未选择文件</div>');
    }
}

$(document).ready(function() {
    // 检查系统状态与当前模型
    checkSystemStatus();
    loadDocumentList();
    // 延迟加载 Ollama 模型，提升首页打开速度
    setTimeout(loadOllamaModels, 2000);
    loadCurrentLLM();
    // 初始化页面持久化聊天记录
    loadChatHistory();

    // 监听文件选择变化，增量添加到全局文件列表
    $('#fileInput').on('change', function() {
        var files = this.files;
        if (files.length > 0) {
            for (var i = 0; i < files.length; i++) {
                // 简单去重，如果同名文件已存在则不添加
                if (!pendingFiles.some(f => f.name === files[i].name)) {
                    pendingFiles.push(files[i]);
                }
            }
            renderPendingFiles();
        }
        // 重置 input 值，允许用户连续选择相同名称的文件（虽然被去重拦截）或分批选择
        $(this).val(''); 
    });

    // 发送消息
    $('#sendBtn').click(function() {
        sendMessage();
    });

    $('#queryInput').keypress(function(e) {
        if(e.which == 13) {
            sendMessage();
        }
    });

    // 监听 LLM 类型下拉框变化
    $('#llmCategorySelect').change(function() {
        var category = $(this).val();
        updateLLMModelOptions(category);
    });

    // 删除文件
    $('#deleteBtn').click(function() {
        var docId = $('#documentSelect').val();
        if (!docId) {
            alert("请选择要删除的文档");
            return;
        }
        
        if (!confirm("确定要删除该文档及其对应的知识库索引吗？")) {
            return;
        }
        
        $('#deleteBtn').prop('disabled', true);
        
        $.ajax({
            url: '/documents/' + docId,
            type: 'DELETE',
            success: function(response) {
                $('#deleteStatus').show().removeClass('alert-danger').addClass('alert-success').text('删除成功');
                checkSystemStatus();
                loadDocumentList();
            },
            error: function(xhr) {
                var err = xhr.responseJSON ? xhr.responseJSON.error : '删除失败';
                $('#deleteStatus').show().removeClass('alert-success').addClass('alert-danger').text(err);
            },
            complete: function() {
                $('#deleteBtn').prop('disabled', false);
                setTimeout(function() {
                    $('#deleteStatus').hide();
                }, 3000);
            }
        });
    });

    // 上传文件(支持多文件)
    $('#uploadForm').on('submit', function(e) {
        e.preventDefault();
        
        if (pendingFiles.length === 0) {
            alert("请选择文件");
            return;
        }
        
        var formData = new FormData();
        for (var i = 0; i < pendingFiles.length; i++) {
            formData.append('files[]', pendingFiles[i]);
        }
        formData.append('db_name', $('#dbNameSelect').val());
        
        $('#uploadBtn').prop('disabled', true);
        $('.spinner-border').removeClass('d-none');
        $('#uploadStatus').hide();
        
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                var msg = '全部上传成功:\n' + response.filenames.join('\n');
                $('#uploadStatus').show().removeClass('alert-danger alert-warning').addClass('alert-success').html(msg.replace(/\n/g, '<br>'));
                checkSystemStatus();
                loadDocumentList();
            },
            error: function(xhr) {
                if (xhr.status === 207) {
                    var res = xhr.responseJSON;
                    var msg = '部分成功:<br>成功: ' + res.uploaded.join(', ') + '<br>失败: ' + res.failed.join('<br>');
                    $('#uploadStatus').show().removeClass('alert-success alert-danger').addClass('alert-warning').html(msg);
                } else {
                    var err = xhr.responseJSON ? xhr.responseJSON.error : '上传失败';
                    $('#uploadStatus').show().removeClass('alert-success alert-warning').addClass('alert-danger').text(err);
                }
            },
            complete: function() {
                $('#uploadBtn').prop('disabled', false);
                $('.spinner-border').addClass('d-none');
                pendingFiles = []; // 上传完成后清空数组
                renderPendingFiles(); // 重新渲染为空
                checkSystemStatus();
                loadDocumentList();
            }
        });
    });

    // 切换模型 (支持全模型)
    $('#switchModelBtn').click(function() {
        var llmType = $('#llmCategorySelect').val();
        var modelName = $('#llmModelSelect').val();
        
        if (!modelName) {
            alert("请选择具体的模型");
            return;
        }
        
        $('#switchModelBtn').prop('disabled', true);
        $('#modelSwitchStatus').text('切换中...').removeClass('text-success text-danger').addClass('text-muted');
        
        $.ajax({
            url: '/llm/set_model',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({llm_type: llmType, model_name: modelName}),
            success: function(res) {
                $('#modelSwitchStatus').text('切换成功').removeClass('text-muted').addClass('text-success');
                $('#currentModelBadge').text(res.current_model);
            },
            error: function(xhr) {
                $('#modelSwitchStatus').text('切换失败').removeClass('text-muted').addClass('text-danger');
            },
            complete: function() {
                $('#switchModelBtn').prop('disabled', false);
                setTimeout(() => $('#modelSwitchStatus').text(''), 3000);
            }
        });
    });
});

let cachedModelsData = {
    ollama_models: [],
    vllm_models: [],
    query_online_models: []
};

function updateLLMModelOptions(category) {
    var select = $('#llmModelSelect');
    select.empty();
    
    var models = [];
    if (category === 'ollama') {
        models = cachedModelsData.ollama_models;
    } else if (category === 'vllm') {
        models = cachedModelsData.vllm_models;
    } else if (category === 'online') {
        models = cachedModelsData.query_online_models;
    }
    
    if (models && models.length > 0) {
        models.forEach(function(model) {
            if (typeof model === 'object') {
                select.append(`<option value="${model.id}">${model.name}</option>`);
            } else {
                select.append(`<option value="${model}">${model}</option>`);
            }
        });
    } else {
        select.append('<option value="">暂无可用模型</option>');
    }
}

function loadCurrentLLM() {
    $.ajax({
        url: '/llm/current',
        type: 'GET',
        success: function(res) {
            $('#currentModelBadge').text(res.current_model);
            var category = res.llm_type;
            // 兼容以前的配置，如果不是 ollama/vllm，则认为是 online
            if (category !== 'ollama' && category !== 'vllm') {
                category = 'online';
            }
            $('#llmCategorySelect').val(category);
            
            // 需要等模型列表加载完成后再选中
            setTimeout(() => {
                updateLLMModelOptions(category);
                if (category === 'online') {
                    $('#llmModelSelect').val(res.actual_id);
                } else if (category === 'ollama') {
                    var modelName = res.current_model.match(/\(([^)]+)\)/)[1];
                    $('#llmModelSelect').val(modelName);
                } else if (category === 'vllm') {
                    var modelName = res.current_model.match(/\(([^)]+)\)/)[1];
                    // vllm options in dropdown have 'vllm: ' prefix usually
                    $('#llmModelSelect').val('vllm: ' + modelName);
                }
            }, 600);
        }
    });
}

function loadOllamaModels() {
    $.ajax({
        url: '/ollama/models',
        type: 'GET',
        success: function(response) {
            cachedModelsData.ollama_models = response.ollama_models || [];
            cachedModelsData.vllm_models = response.vllm_models || [];
            cachedModelsData.query_online_models = response.query_online_models || [];
            
            var currentCategory = $('#llmCategorySelect').val();
            updateLLMModelOptions(currentCategory);
        },
        error: function() {
            console.error('获取模型列表失败');
        }
    });
}

// 全局变量用于存储当前的请求控制器
let currentAbortController = null;
let currentChatSessionId = null;

// 配置 marked.js
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,
        gfm: true
    });
}

function sendMessage() {
    var query = $('#queryInput').val().trim();
    if (!query) return;

    // 获取选中的数据库
    var selectedDbs = [];
    $('.db-checkbox:checked').each(function() {
        selectedDbs.push($(this).val());
    });
    
    // 获取工具开关状态
    var enableTools = $('#enableToolsToggle').is(':checked');
    
    if (selectedDbs.length === 0) {
        alert("请至少选择一个查询库！");
        return;
    }

    // 添加用户消息
    var displayQuery = query + '\n<small style="opacity: 0.8; font-size: 0.8em;">(查询范围: ' + selectedDbs.join(', ') + ')</small>';
    appendMessage('user', displayQuery);
    $('#queryInput').val('');

    // 显示加载中并准备流式接收的空气泡
    var loadingId = appendLoading();
    var botMsgId = 'bot-msg-' + Date.now();
    
    // 切换按钮状态
    $('#sendBtn').addClass('d-none');
    $('#stopBtn').removeClass('d-none');

    currentAbortController = new AbortController();

    fetch('/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            question: query,
            db_names: selectedDbs,
            enable_tools: enableTools
        }),
        signal: currentAbortController.signal
    }).then(async response => {
        removeLoading(loadingId);
        
        // 创建一个空的回复气泡
        appendEmptyBotMessage(botMsgId);
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let done = false;

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
                const chunkStr = decoder.decode(value, {stream: true});
                const lines = chunkStr.split('\n');
                
                for (let line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        if (data.type === 'chunk') {
                            appendChunkToMessage(botMsgId, data.content);
                        } else if (data.type === 'action') {
                            $(`#${botMsgId}-text`).append(data.content);
                            scrollToBottom();
                        } else if (data.type === 'sources') {
                            appendSourcesToMessage(botMsgId, data.sources);
                        }
                    } catch (e) {
                        console.error('Error parsing JSON stream chunk:', line);
                    }
                }
            }
        }
    }).catch(err => {
        if (err.name === 'AbortError') {
            appendChunkToMessage(botMsgId, '\n[已终止输出]');
        } else {
            removeLoading(loadingId);
            appendMessage('bot', '抱歉，系统出现错误，请稍后再试。');
        }
    }).finally(() => {
        // 恢复按钮状态
        $('#sendBtn').removeClass('d-none');
        $('#stopBtn').addClass('d-none');
        currentAbortController = null;

        // 渲染打分组件
        appendRatingToMessage(botMsgId, query);
        
        // 渲染完成后稍微延迟一下再保存，确保 DOM 已经完全更新
        setTimeout(() => {
            scrollToBottom();
            saveChatHistory();
        }, 100);
    });
}

// 绑定终止按钮事件
$(document).ready(function() {
    $('#stopBtn').click(function() {
        if (currentAbortController) {
            currentAbortController.abort();
        }
    });
});

function appendEmptyBotMessage(msgId) {
    var html = `
        <div class="d-flex flex-row justify-content-start mb-3" id="${msgId}-container">
            <div class="chat-bubble bot markdown-body" style="position: relative; width: 100%;">
                <details class="mb-2" id="${msgId}-thinking-container" style="display: none; background-color: #f8f9fa; border-radius: 6px; padding: 5px 10px; border: 1px solid #e9ecef;">
                    <summary style="cursor: pointer; color: #6c757d; font-size: 0.9em; font-weight: bold; user-select: none;">
                        🧠 思考过程 <span class="spinner-border spinner-border-sm ms-1" role="status" aria-hidden="true" style="width: 0.8rem; height: 0.8rem; vertical-align: text-bottom;"></span>
                    </summary>
                    <div class="mt-2 text-muted" id="${msgId}-thinking-text" style="font-size: 0.85em; white-space: pre-wrap; max-height: 300px; overflow-y: auto;"></div>
                    <div id="${msgId}-sources-thinking" class="mt-2"></div>
                </details>
                <div class="mb-0 message-content" id="${msgId}-text"></div>
                <div id="${msgId}-sources" style="display: none;"></div>
                <div id="${msgId}-rating" class="mt-2 text-end" style="font-size: 0.85rem; border-top: 1px dashed #dee2e6; padding-top: 5px;"></div>
            </div>
        </div>
    `;
    $('#chatHistory').append(html);
    scrollToBottom();
}

function appendRatingToMessage(msgId, question) {
    var ratingContainer = $(`#${msgId}-rating`);
    var fullAnswerText = $(`#${msgId}-text`).text();
    
    // 生成星星 HTML 结构
    var html = `
        <div class="rating-stars" data-msgid="${msgId}" data-question="${question}" style="cursor: pointer; color: #ffc107; font-size: 1.2rem;">
            <span class="text-muted me-2" style="font-size: 0.85rem;">评价此回答：</span>
            <span class="star" data-value="1">★</span>
            <span class="star" data-value="2">★</span>
            <span class="star" data-value="3">★</span>
            <span class="star" data-value="4">★</span>
            <span class="star" data-value="5">★</span>
            <span class="rating-text text-muted ms-2" style="font-size: 0.8rem; display: none;"></span>
        </div>
    `;
    ratingContainer.html(html);
    
    var stars = $(`#${msgId}-rating .star`);
    var container = $(`#${msgId}-rating .rating-stars`);
    
    // 悬浮效果
    stars.hover(
        function() {
            if (container.hasClass('submitted')) return;
            var val = $(this).data('value');
            stars.each(function() {
                $(this).text($(this).data('value') <= val ? '★' : '☆');
            });
        },
        function() {
            if (container.hasClass('submitted')) return;
            // 默认显示5星
            stars.text('★');
        }
    );
    
    // 点击打分直接提交
    stars.click(function() {
        if (container.hasClass('submitted')) return;
        
        var score = $(this).data('value');
        var q = container.data('question');
        var a = fullAnswerText;
        
        // 锁定状态
        container.addClass('submitted');
        stars.each(function() {
            $(this).text($(this).data('value') <= score ? '★' : '☆').css('cursor', 'default');
        });
        $(`#${msgId}-rating .rating-text`).text(`已评 ${score} 分`).show();
        
        // 如果低于4分触发重新分析
        if (parseInt(score) < 4) {
            triggerReanalyze(q, a, score);
        }
    });
}

function triggerReanalyze(question, previous_answer, score) {
    // 显示加载中并准备流式接收的空气泡
    var loadingId = appendLoading();
    var botMsgId = 'bot-msg-reanalyze-' + Date.now();
    
    // 切换按钮状态
    $('#sendBtn').addClass('d-none');
    $('#stopBtn').removeClass('d-none');

    currentAbortController = new AbortController();

    fetch('/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            question: question,
            answer: previous_answer,
            score: score
        }),
        signal: currentAbortController.signal
    }).then(async response => {
        removeLoading(loadingId);
        appendEmptyBotMessage(botMsgId);
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let done = false;

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
                const chunkStr = decoder.decode(value, {stream: true});
                const lines = chunkStr.split('\n');
                for (let line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        if (data.type === 'chunk') {
                            appendChunkToMessage(botMsgId, data.content);
                        } else if (data.type === 'action') {
                            $(`#${botMsgId}-text`).append(data.content);
                            scrollToBottom();
                        } else if (data.type === 'sources') {
                            appendSourcesToMessage(botMsgId, data.sources);
                        }
                    } catch (e) {}
                }
            }
        }
    }).finally(() => {
        $('#sendBtn').removeClass('d-none');
        $('#stopBtn').addClass('d-none');
        currentAbortController = null;

        // 流结束时，解析内联引用
        var textElem = $(`#${botMsgId}-text`);
        textElem.html(formatCitations(textElem.html()));
        
        // 渲染完成后稍微延迟一下再保存，确保 DOM 已经完全更新
        setTimeout(() => {
            scrollToBottom();
            saveChatHistory();
        }, 100);
    });
}

function executeAutoScript(scriptName) {
    appendMessage('user', '执行自动化脚本: ' + scriptName);
    var loadingId = appendLoading();
    
    $.ajax({
        url: '/api/run_script',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ script_name: scriptName }),
        success: function(res) {
            removeLoading(loadingId);
            appendMessage('bot', res.message);
        },
        error: function(err) {
            removeLoading(loadingId);
            appendMessage('bot', '脚本执行失败: ' + err.responseText);
        }
    });
}

function appendChunkToMessage(msgId, textChunk) {
    var textElem = $(`#${msgId}-text`);
    
    // 判断是否为思考过程
    if (textChunk.startsWith('> ') || textChunk.includes('思考过程')) {
        var thinkingContainer = $(`#${msgId}-thinking-container`);
        var thinkingText = $(`#${msgId}-thinking-text`);
        
        if (thinkingContainer.css('display') === 'none') {
            thinkingContainer.show();
            // 默认展开
            thinkingContainer.prop('open', true);
        }
        
        // 追加思考内容并渲染
        var currentThinking = thinkingText.attr('data-raw') || '';
        currentThinking += textChunk;
        thinkingText.attr('data-raw', currentThinking);
        thinkingText.html(marked.parse(currentThinking));
        
        // 思考过程结束的标志
        if (textChunk.includes('开始整合知识生成最终分析报告') || textChunk.includes('开始整合上下文并生成最终分析报告')) {
            // 移除 spinner
            thinkingContainer.find('.spinner-border').remove();
            // 思考结束后自动折叠
            setTimeout(() => {
                thinkingContainer.prop('open', false);
            }, 1000);
        }
        scrollToBottom();
        return;
    }

    var rawText = textElem.data('raw') || '';
    rawText += textChunk;
    textElem.data('raw', rawText);
    
    try {
        // Format citations before parsing markdown
        var formattedText = formatCitations(rawText);
        
        var parsedHtml = formattedText;
        if (typeof marked !== 'undefined') {
            if (typeof marked.parse === 'function') {
                parsedHtml = marked.parse(formattedText);
            } else if (typeof marked === 'function') {
                parsedHtml = marked(formattedText);
            } else {
                parsedHtml = formattedText.replace(/\n/g, '<br>');
            }
        } else {
            parsedHtml = formattedText.replace(/\n/g, '<br>');
        }
        textElem.html(parsedHtml);
        
        // Render math formulas if KaTeX is available
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(textElem[0], {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
                throwOnError: false
            });
        }
        
        // Apply syntax highlighting
        textElem.find('pre code').each(function(i, block) {
            if (typeof hljs !== 'undefined') {
                hljs.highlightElement(block);
            }
        });
    } catch (err) {
        console.error("Error rendering message chunk:", err);
        // Fallback: display raw text
        var safeChunk = rawText.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
        textElem.html(safeChunk);
    }
    
    scrollToBottom();
}

function appendSourcesToMessage(msgId, sources) {
    var sourcesHtml = renderSources(sources);
    
    // 如果有思考过程的容器，优先追加到思考过程里
    var thinkingSourcesElem = $(`#${msgId}-sources-thinking`);
    if (thinkingSourcesElem.length > 0) {
        thinkingSourcesElem.html(sourcesHtml);
    } else {
        // 后备：附加到原来的位置
        var sourcesElem = $(`#${msgId}-sources`);
        sourcesElem.html(sourcesHtml).show();
    }
    
    scrollToBottom();
}

// Helper function to format citations like [1], [2] into clickable or styled elements
function formatCitations(text) {
    if (!text) return text;
    // Replace [数字] with a styled span that stands out
    return text.replace(/\[(\d+)\]/g, function(match, p1) {
        return `<span class="badge bg-info text-dark citation-badge" style="cursor:help; margin-left:2px; font-size:0.75em; vertical-align:super;" title="参考来源 ${p1}">[${p1}]</span>`;
    });
}

function appendMessage(sender, text, sources) {
    var bubbleClass = sender === 'user' ? 'user' : 'bot';
    var alignClass = sender === 'user' ? 'justify-content-end' : 'justify-content-start';
    
    var displayContent = text;
    if (sender !== 'user') {
        try {
            var formattedText = formatCitations(text);
            if (typeof marked !== 'undefined') {
                if (typeof marked.parse === 'function') {
                    displayContent = marked.parse(formattedText);
                } else if (typeof marked === 'function') {
                    displayContent = marked(formattedText);
                } else {
                    displayContent = formattedText.replace(/\n/g, '<br>');
                }
            } else {
                displayContent = formattedText.replace(/\n/g, '<br>');
            }
        } catch (err) {
            console.error("Error formatting message:", err);
            displayContent = text.replace(/\n/g, '<br>');
        }
    } else {
        displayContent = text.replace(/\n/g, '<br>');
    }
    
    var html = `
        <div class="d-flex flex-row ${alignClass} mb-3">
            <div class="chat-bubble ${bubbleClass} markdown-body" style="max-width: 80%;">
                <div class="mb-0 message-content" data-raw="${text.replace(/"/g, '&quot;')}">${displayContent}</div>
                ${renderSources(sources)}
            </div>
        </div>
    `;
    
    var newElem = $(html);
    $('#chatHistory').append(newElem);
    
    // Apply KaTeX and highlight.js for bot messages
    if (sender !== 'user') {
        var contentElem = newElem.find('.message-content')[0];
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(contentElem, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
                throwOnError: false
            });
        }
        if (typeof hljs !== 'undefined') {
            newElem.find('pre code').each(function(i, block) {
                hljs.highlightElement(block);
            });
        }
    }
    
    scrollToBottom();
    saveChatHistory();
}

function saveChatHistory() {
    var chatContent = $('#chatHistory').html();
    localStorage.setItem('rag_chat_history', chatContent);
}

function loadChatHistory() {
    var chatContent = localStorage.getItem('rag_chat_history');
    if (chatContent) {
        $('#chatHistory').html(chatContent);
        // 恢复所有消息的 marked 渲染状态
        $('#chatHistory .message-content').each(function() {
            var rawText = $(this).attr('data-raw');
            if (rawText) {
                // 如果是用户消息或者没有被渲染过的内容，才重新渲染
                // 实际上保存的时候已经是渲染好的 HTML 了，所以这一步可能不是必须的
                // 但如果发现有未渲染的部分，可以在这里统一处理
            }
        });
        
        // 重新绑定思考过程的折叠事件（如果保存时被写死了）
        $('#chatHistory details').each(function() {
            var summary = $(this).find('summary');
            // 确保 summary 存在且没有绑定过额外的自定义事件
        });

        // 重新渲染数学公式和代码高亮（因为 HTML 是静态存入的，可能丢失一些动态绑定的事件或样式类）
        if (typeof renderMathInElement !== 'undefined') {
            $('#chatHistory .message-content').each(function() {
                renderMathInElement(this, {
                    delimiters: [
                        {left: "$$", right: "$$", display: true},
                        {left: "\\[", right: "\\]", display: true},
                        {left: "$", right: "$", display: false},
                        {left: "\\(", right: "\\)", display: false}
                    ]
                });
            });
        }
        
        scrollToBottom();
    } else {
        clearChat(false); // 不触发 saveChatHistory
    }
}

function renderSources(sources) {
    if (!sources || sources.length === 0) return '';
    
    var html = '<div class="mt-2 pt-2 border-top">';
    html += '<small class="text-muted d-block mb-2">参考资料：</small>';
    html += '<div class="d-flex flex-wrap gap-2">'; // 横向排布容器
    
    sources.forEach(function(src, index) {
        html += `
            <div class="source-item position-relative">
                <span class="badge bg-secondary source-badge" onclick="toggleSource(this)" style="cursor: pointer;">来源 ${index + 1}</span>
                <div class="source-content position-absolute bg-white border rounded shadow p-2" style="display: none; z-index: 1000; width: 300px; max-height: 200px; overflow-y: auto; left: 0; top: 100%; margin-top: 5px;">
                    <div class="mb-2" style="font-size: 0.85rem;">${src.content}</div>
                    <div class="text-end text-muted" style="font-size: 0.75rem;">DocName: ${src.doc_name || '-'} | DocID: ${src.doc_id || '-'}</div>
                </div>
            </div>
        `;
    });
    
    html += '</div></div>';
    return html;
}

function toggleSource(element) {
    $(element).next('.source-content').slideToggle();
}

function appendLoading() {
    var id = 'loading-' + Date.now();
    var html = `
        <div id="${id}" class="d-flex flex-row justify-content-start mb-3">
            <div class="chat-bubble bot">
                <div class="spinner-grow spinner-grow-sm text-secondary" role="status"></div>
                <div class="spinner-grow spinner-grow-sm text-secondary" role="status"></div>
                <div class="spinner-grow spinner-grow-sm text-secondary" role="status"></div>
            </div>
        </div>
    `;
    $('#chatHistory').append(html);
    scrollToBottom();
    return id;
}

function removeLoading(id) {
    $('#' + id).remove();
}

function scrollToBottom() {
    var chatHistory = document.getElementById("chatHistory");
    // 加一点冗余高度，确保如果有 margin 折叠也能滚到底
    chatHistory.scrollTop = chatHistory.scrollHeight + 100;
}

function clearChat(save = true) {
    $('#chatHistory').html(`
        <div class="d-flex flex-row justify-content-start mb-3">
            <div class="p-3 bg-white border rounded shadow-sm" style="max-width: 80%;">
                <p class="mb-0">您好！我是湖北移动运维助手。请问有什么故障需要排查？</p>
            </div>
        </div>
    `);
    if (save) {
        saveChatHistory();
    }
}

function checkSystemStatus() {
    $.ajax({
        url: '/status',
        type: 'GET',
        success: function(response) {
            $('#docCount').text(response.doc_count || 0);
            $('#indexStatus').text(response.status === 'ok' ? '正常' : '异常');
        }
    });
}

function loadDocumentList() {
    $.ajax({
        url: '/documents',
        type: 'GET',
        success: function(response) {
            var select = $('#documentSelect');
            select.empty();
            select.append('<option value="" selected>请选择要删除的文档...</option>');
            
            if (response.documents && response.documents.length > 0) {
                $('#docCount').text(response.documents.length);
                response.documents.forEach(function(doc) {
                    var statusStr = doc.status === 'completed' ? '已解析' : 
                                  (doc.status === 'failed' ? '解析失败' : '处理中');
                    var dbLabel = doc.db_name ? `[${doc.db_name}]` : '[default]';
                    select.append(`<option value="${doc.doc_id}">${dbLabel} ${doc.doc_name} (${statusStr})</option>`);
                });
            } else {
                $('#docCount').text('0');
            }
        },
        error: function(xhr) {
            console.error("Failed to load document list", xhr);
        }
    });
}

function clearSystemData() {
    if (!confirm("警告：此操作将清空所有数据库记录和向量索引，确定要继续吗？")) {
        return;
    }
    
    $.ajax({
        url: '/documents/clear_all',
        type: 'DELETE',
        success: function(response) {
            alert('所有数据已成功清空');
            checkSystemStatus();
            loadDocumentList();
        },
        error: function(xhr) {
            alert('清空失败: ' + (xhr.responseJSON ? xhr.responseJSON.error : '未知错误'));
        }
    });
}
