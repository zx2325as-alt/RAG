// 全局变量用于存储待上传的文件列表，避免页面切换或多次选择丢失
let pendingFiles = [];

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
    renderPendingFiles(); // 初始化文件列表

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
    $('#llmTypeSelect').change(function() {
        if ($(this).val() === 'ollama') {
            $('#ollamaModelGroup').show();
        } else {
            $('#ollamaModelGroup').hide();
        }
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
        var llmType = $('#llmTypeSelect').val();
        var modelName = $('#ollamaModelSelect').val();
        
        if (llmType === 'ollama' && !modelName) {
            alert("请选择具体的 Ollama 模型");
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

function loadCurrentLLM() {
    $.ajax({
        url: '/llm/current',
        type: 'GET',
        success: function(res) {
            $('#currentModelBadge').text(res.current_model);
            $('#llmTypeSelect').val(res.llm_type).trigger('change');
            if (res.llm_type === 'ollama') {
                setTimeout(() => { $('#ollamaModelSelect').val(res.current_model); }, 500);
            }
        }
    });
}

function loadOllamaModels() {
    $.ajax({
        url: '/ollama/models',
        type: 'GET',
        success: function(response) {
            var select = $('#ollamaModelSelect');
            select.empty();
            if (response.models && response.models.length > 0) {
                response.models.forEach(function(model) {
                    select.append(`<option value="${model}">${model}</option>`);
                });
            } else {
                select.append('<option value="">未发现本地模型</option>');
            }
        },
        error: function() {
            var select = $('#ollamaModelSelect');
            select.empty();
            select.append('<option value="">Ollama 服务未启动</option>');
        }
    });
}

// 全局变量用于存储当前的请求控制器
let currentAbortController = null;

function sendMessage() {
    var query = $('#queryInput').val().trim();
    if (!query) return;

    // 获取选中的数据库
    var selectedDbs = [];
    $('.db-checkbox:checked').each(function() {
        selectedDbs.push($(this).val());
    });
    
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
            db_names: selectedDbs
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
            <div class="chat-bubble bot" style="position: relative;">
                <p class="mb-0" id="${msgId}-text"></p>
                <div id="${msgId}-sources"></div>
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
    // 将 \n 转换为 <br> 但要防止 XSS，这里简单处理
    var safeChunk = textChunk.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
    textElem.append(safeChunk);
    scrollToBottom();
}

function appendSourcesToMessage(msgId, sources) {
    var sourcesElem = $(`#${msgId}-sources`);
    sourcesElem.html(renderSources(sources));
    scrollToBottom();
}

function appendMessage(sender, text, sources) {
    var bubbleClass = sender === 'user' ? 'user' : 'bot';
    var alignClass = sender === 'user' ? 'justify-content-end' : 'justify-content-start';
    
    var html = `
        <div class="d-flex flex-row ${alignClass} mb-3">
            <div class="chat-bubble ${bubbleClass}">
                <p class="mb-0">${text.replace(/\n/g, '<br>')}</p>
                ${renderSources(sources)}
            </div>
        </div>
    `;
    
    $('#chatHistory').append(html);
    scrollToBottom();
}

function renderSources(sources) {
    if (!sources || sources.length === 0) return '';
    
    var html = '<div class="mt-2 pt-2 border-top">';
    html += '<small class="text-muted d-block mb-1">参考资料：</small>';
    
    sources.forEach(function(src, index) {
        html += `
            <div class="mb-1">
                <span class="badge bg-secondary source-badge" onclick="toggleSource(this)">来源 ${index + 1}</span>
                <div class="source-content">
                    ${src.content}
                    <div class="mt-1 text-end"><small>DocID: ${src.doc_id || '-'}</small></div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
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
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function clearChat() {
    $('#chatHistory').html(`
        <div class="d-flex flex-row justify-content-start mb-3">
            <div class="p-3 bg-white border rounded shadow-sm" style="max-width: 80%;">
                <p class="mb-0">您好！我是湖北移动运维助手。请问有什么故障需要排查？</p>
            </div>
        </div>
    `);
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
