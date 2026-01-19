// 录音控制器


$(function () {
    /* ========= 全局运行状态 ========= */
    let ws = null;                  // WebSocket 实例
    let recorder = null;            // MediaRecorder 实例
    let sendTimer = null;           // 音频发送定时器
    let isRecording = false;     // 当前是否处于录音状态

    /* ========= DOM 引用 ========= */
    const $recordBtn = $('#recordButton');
    const $result = $('#transcriptionResult');
    const $summary = $('#summaryResult');

    /* ========= 配置项（集中管理） ========= */
    const portNum = 27000;         // 服务器端口,请根据实际情况修改

    /* ========= 按钮点击 ========= */
    $recordBtn.on('click', function () {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    /* ========= 开始录制 ========= */
    function startRecording() {
        const lang = $('#lang').val() || 'auto';
        const sv = $('#speakerVerification').is(':checked') ? 1 : 0;

        const wsUrl = `ws://localhost:${portNum}/ws/transcribe?lang=${lang}&sv=${sv}`;
        ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';

        // 清空之前的结果
        $result.text('');
        $summary.text('');

        ws.onopen = function () {
            console.log('WebSocket connected');

            recorder.start();

            // 每500ms发送一次音频数据
            sendTimer = setInterval(function () {
                const blob = recorder.getBlob();
                if (blob && ws.readyState === WebSocket.OPEN) {
                    ws.send(blob);
                }
                recorder.clear();
            }, 500);
        };

        /* ========= 毫秒转时间戳 ========= */
        function msToTimestamp(ms, withHour = false) {
            const totalSeconds = Math.floor(ms / 1000);
            const milliseconds = ms % 1000;

            const seconds = totalSeconds % 60;
            const minutes = Math.floor(totalSeconds / 60) % 60;
            const hours = Math.floor(totalSeconds / 3600);

            if (withHour || hours > 0) {
                return (
                    String(hours).padStart(2, "0") + ":" +
                    String(minutes).padStart(2, "0") + ":" +
                    String(seconds).padStart(2, "0") + "." +
                    String(milliseconds).padStart(3, "0")
                );
            } else {
                return (
                    String(minutes).padStart(2, "0") + ":" +
                    String(seconds).padStart(2, "0") + "." +
                    String(milliseconds).padStart(3, "0")
                );
            }
        }


        /* ========= 清理ASR文本中的标签 ========= */
        function cleanAsrText(text) {
            return text.replace(/<\|[^]+?\|>/g, "").trim();
        }

        /* ========= 格式化字幕行 ========= */
        function formatSubtitle(info, rawText) {
            const start = msToTimestamp(info.start_ms);
            const end = msToTimestamp(info.end_ms);
            const text = cleanAsrText(rawText);

            return `[${start} - ${end}] ${text}`;
        }


        ws.onmessage = function (evt) {
            try {
                const res = JSON.parse(evt.data);

                // 1) 实时字幕消息
                if (res.code === 0 && res.data) {
                    const info = JSON.parse(res.info);
                    appendText($result, formatSubtitle(info, res.data));
                }

                // 2) 最终 JSON
                if (res.code === 1 && res.data) {
                    // res.data 是 final_obj 的 JSON 字符串
                    // 你要显示到转录框，就解析 segments 后渲染
                    try {
                        const finalObj = JSON.parse(res.data);
                        const segs = finalObj.segments || [];
                        const lines = segs.map(s => {
                            const start = msToTimestamp(s.start_ms || 0);
                            const end = msToTimestamp(s.end_ms || 0);
                            const spk = s.spk ?? "spk";
                            const text = cleanAsrText((s.text || "").trim());
                            return `[${start} - ${end}] ${spk}: ${text}`;
                        });
                        replaceText($result, lines.join("\n"));
                    } catch (e) {
                        replaceText($result, res.data || "");
                    }
                    return;
                }

                // 3) summary 单独消息
                if (res.code === 2 && res.data) {
                    replaceText($summary, res.data || "");
                    // summary 收到后再关闭 ws（如果你希望后端发完再关）
                    try {
                        ws.close();
                        $recordBtn.prop('disabled', false);// 重新启用按钮
                    } catch (e) {
                    }
                }
            } catch (e) {
                appendText($summary, "收到异常消息，可能是summary部分：");
                appendText($summary, evt.data);
            }
        };

        ws.onclose = function () {
            console.log('WebSocket closed');

            // 这里只做UI复位，不要再调用 stopRecording（否则可能重复发 end）
            if (sendTimer) {
                clearInterval(sendTimer);
                sendTimer = null;
            }
            $recordBtn
                .removeClass('recording')
                .text('开始录制');

            isRecording = false;
        };


        ws.onerror = function (err) {
            console.error('WebSocket error', err);
        };

        $recordBtn
            .addClass('recording')
            .text('停止录制');

        isRecording = true;
    }

    /* ========= 停止录制 ========= */
    function stopRecording() {
        // 暂时关闭按钮，防止重复点击
        $recordBtn
            .removeClass('recording')
            .text("处理中...")
            .prop('disabled', true);


        if (sendTimer) {
            clearInterval(sendTimer);
            sendTimer = null;
        }

        if (recorder) {
            recorder.stop();
        }

        // 不立刻 close ws
        // 改为发送一个 end 信号，让后端在同一条连接里跑离线并返回 code=1
        if (ws && ws.readyState === WebSocket.OPEN) {
            try {
                ws.send(JSON.stringify({type: "end"}));
            } catch (e) {
                console.warn("send end failed", e);
            }
        }

        isRecording = false;
    }


    /* ========= 追加文本 ========= */
    function appendText($el, text) {
        const el = $el[0]; // 取 DOM 元素

        // 是否接近底部（10px 容差）
        const atBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) < 10;

        // 追加文本（保留换行）
        $el.text($el.text() + text + "\n");

        // 如果用户原本在底部，就自动滚到底
        if (atBottom) {
            el.scrollTop = el.scrollHeight; // 用原生最稳
        }
    }

    /* ========= 替换文本 ========= */
    function replaceText($el, text) {
        // 替换文本（保留换行）
        $el.text(text + "\n");
    }

    /* ========= 初始化录音 ========= */
    navigator.getUserMedia = navigator.getUserMedia || navigator.mediaDevices.getUserMedia;

    // 浏览器兼容性检查
    if (!navigator.getUserMedia) {
        alert('浏览器不支持录音');
        return;
    }

    // 获取麦克风权限并初始化 Recorder
    navigator.getUserMedia(
        {audio: true},
        function (stream) {
            recorder = new Recorder(stream);
        },
        function (err) {
            console.error('获取麦克风失败', err);
        }
    );

    /* ========= Recorder 定义 ========= */
    function Recorder(stream) {
        const context = new (window.AudioContext || window.webkitAudioContext)();
        const input = context.createMediaStreamSource(stream);
        const processor = context.createScriptProcessor(4096, 1, 1);

        let buffer = [];
        let size = 0;

        processor.onaudioprocess = function (e) {
            const data = e.inputBuffer.getChannelData(0);
            const downsampled = downsample(data, context.sampleRate, 16000);
            buffer.push(downsampled);
            size += downsampled.length;
        };

        input.connect(processor);
        processor.connect(context.destination);

        this.start = function () {
            buffer = [];
            size = 0;
        };

        this.stop = function () {
            processor.disconnect();
            input.disconnect();
        };

        this.clear = function () {
            buffer = [];
            size = 0;
        };

        this.getBlob = function () {
            if (!size) return null;

            const pcm = new Int16Array(size);
            let offset = 0;

            buffer.forEach(chunk => {
                for (let i = 0; i < chunk.length; i++) {
                    pcm[offset++] = Math.max(-1, Math.min(1, chunk[i])) * 0x7fff;
                }
            });

            return new Blob([pcm.buffer], {type: 'audio/pcm'});
        };
    }

    /* ========= 重采样 ========= */
    function downsample(buffer, inRate, outRate) {
        if (outRate === inRate) return buffer;
        const ratio = inRate / outRate;
        const newLen = Math.round(buffer.length / ratio);
        const result = new Float32Array(newLen);

        let offset = 0;
        for (let i = 0; i < newLen; i++) {
            const start = Math.round(i * ratio);
            const end = Math.round((i + 1) * ratio);
            let sum = 0;
            let count = 0;

            for (let j = start; j < end && j < buffer.length; j++) {
                sum += buffer[j];
                count++;
            }
            result[offset++] = sum / count;
        }
        return result;
    }

});