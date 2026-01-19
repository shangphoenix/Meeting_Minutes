// download button 功能实现
// 下载结果和摘要的按钮逻辑
// 需要放在最后加载的 js 里，以确保结果区域已经存在

$(function () {
    const TIMEOUT_MS = 1500;
    const btnResDownload = document.getElementById("btnResDownload");
    const btnSummaryDownload = document.getElementById("btnSummaryDownload");

    const summaryEl = document.getElementById("summaryResult");
    const resultEl = document.getElementById("transcriptionResult");

    // 如果两个按钮都不存在，直接退出
    if (!btnResDownload && !btnSummaryDownload) return;

    const labelRes = btnResDownload ? btnResDownload.querySelector("span.button") : null;
    const labelSum = btnSummaryDownload ? btnSummaryDownload.querySelector("span.button") : null;

    const ORI_RES = labelRes ? labelRes.textContent : "下载";
    const ORI_SUM = labelSum ? labelSum.textContent : "下载";

    let resTimer = null;
    let sumTimer = null;

    // 替换文件名中的非法字符
    function sanitizeFilename(name) {
        return String(name).replace(/[\\/:*?"<>|]/g, "_");
    }

    // 生成时间戳字符串，格式：YYYYMMDD_HHMMSS
    function makeTimestamp() {
        const d = new Date();
        const pad = (n) => String(n).padStart(2, "0");
        return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
    }

    // 设置按钮状态
    function setState(btn, label, text, disabled) {
        if (label) label.textContent = text;
        if (btn) btn.disabled = !!disabled;
    }

    // 恢复按钮状态
    function restore(which) {
        if (which === "res") {
            if (resTimer) clearTimeout(resTimer);
            resTimer = setTimeout(() => {
                setState(btnResDownload, labelRes, ORI_RES, false);
                resTimer = null;
            }, TIMEOUT_MS);
        } else {
            if (sumTimer) clearTimeout(sumTimer);
            sumTimer = setTimeout(() => {
                setState(btnSummaryDownload, labelSum, ORI_SUM, false);
                sumTimer = null;
            }, TIMEOUT_MS);
        }
    }

    // 使用 File System Access API 弹出保存文件对话框
    async function saveTextAsTxtDialog(suggestedName, text) {
        if (!window.showSaveFilePicker) return false; // 不支持

        const handle = await window.showSaveFilePicker({
            suggestedName: sanitizeFilename(suggestedName),
            types: [
                {
                    description: "文本文件",
                    accept: {"text/plain": [".txt"]},
                },
            ],
        });

        const writable = await handle.createWritable();
        await writable.write(text); // 直接写文本
        await writable.close();
        return true;
    }

    // 备用下载方法，创建隐藏链接下载
    function downloadFallback(filename, text) {
        const blob = new Blob([text], {type: "text/plain;charset=utf-8"});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = sanitizeFilename(filename);
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    }

    // 处理下载逻辑
    async function handleDownload(which) {
        const isRes = which === "res";
        const btn = isRes ? btnResDownload : btnSummaryDownload;
        const label = isRes ? labelRes : labelSum;

        const text = isRes
            ? (resultEl ? resultEl.textContent : "")
            : (summaryEl ? summaryEl.textContent : "");

        if (!btn || !text.trim()) return;

        setState(btn, label, "保存中...", true);

        const ts = makeTimestamp();
        const filename = isRes ? `result_${ts}.txt` : `summary_${ts}.txt`;

        try {
            const ok = await saveTextAsTxtDialog(filename, text);
            if (!ok) {
                // 不支持“另存为弹窗” -> 普通下载
                downloadFallback(filename, text);
                setState(btn, label, "已下载", true);
            } else {
                setState(btn, label, "已保存", true);
            }
            restore(which);
        } catch (e) {
            // 用户取消也会抛 AbortError
            setState(btn, label, "已取消", true);
            restore(which);
        }
    }

    // ====== 绑定两个下载按钮 ======
    if (btnResDownload) btnResDownload.onclick = () => handleDownload("res");
    if (btnSummaryDownload) btnSummaryDownload.onclick = () => handleDownload("sum");
})();
