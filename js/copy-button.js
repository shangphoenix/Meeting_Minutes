// copy button 功能实现
// 复制结果和摘要的按钮逻辑

$(function () {
    const TIMEOUT_MS = 1500;

    // 设置按钮状态
    function setState(btn, labelEl, text, disabled) {
        if (labelEl) labelEl.textContent = text;
        if (btn) btn.disabled = !!disabled;
    }

    function bindCopyButton({btnId, getText, okText = "已复制", failText = "复制失败"}) {
        const btn = document.getElementById(btnId);
        if (!btn) return;

        const labelEl = btn.querySelector("span.button");
        const ORI_TEXT = labelEl ? labelEl.textContent : "复制";

        let timer = null;

        btn.onclick = () => {
            const textToCopy = getText ? String(getText() || "") : "";
            if (!textToCopy.trim()) return;

            navigator.clipboard.writeText(textToCopy).then(() => {
                if (timer) clearTimeout(timer);

                setState(btn, labelEl, okText, true);

                timer = setTimeout(() => {
                    setState(btn, labelEl, ORI_TEXT, false);
                    timer = null;
                }, TIMEOUT_MS);
            }).catch(() => {
                if (timer) clearTimeout(timer);

                // 失败时一般不强制禁用按钮，避免用户想立刻再试
                setState(btn, labelEl, failText, false);

                timer = setTimeout(() => {
                    setState(btn, labelEl, ORI_TEXT, false);
                    timer = null;
                }, TIMEOUT_MS);
            });
        };
    }

    // ====== 绑定两个复制按钮 ======
    bindCopyButton({
        btnId: "btnResCopy",
        getText: () => $result.text(),
    });

    bindCopyButton({
        btnId: "btnSummaryCopy",
        getText: () => $summary.text(),
    });
});
