import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

app = FastAPI()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    all_texts = []
    segments = []
    seq = 0

    try:
        while True:
            msg = await ws.receive()

            # ---- 1) 前端发文本(JSON)控制指令 ----
            if msg.get("text") is not None:
                text = msg["text"]

                # 允许纯文本 stop 或 JSON {"type":"stop"}
                try:
                    cmd = json.loads(text)
                except Exception:
                    cmd = {"type": text}

                if cmd.get("type") in ("stop", "final", "end"):
                    raw_text = "\n".join(all_texts).strip()

                    # 这里用假 summary 做测试（真实项目换成 deepseek_optimize）
                    summary_text = f"【TEST SUMMARY】共收到 {len(all_texts)} 段文本。"

                    await ws.send_json({
                        "code": 3,
                        "info": "final_summary",
                        "summary": summary_text,
                        "result": {
                            "raw_text": raw_text,
                            "segments": segments,
                        }
                    })

                    await ws.close()
                    break

                # 也允许前端直接发一段“文本结果”，模拟 ASR 输出
                if cmd.get("type") == "text":
                    t = cmd.get("text", "")
                    if t:
                        all_texts.append(t)
                        seg = {"id": seq, "text": t}
                        segments.append(seg)
                        seq += 1

                        # 模拟实时推送
                        await ws.send_json({
                            "code": 0,
                            "info": "partial",
                            "result": seg
                        })
                continue

            # ---- 2) 前端发二进制（模拟音频） ----
            if msg.get("bytes") is not None:
                data = msg["bytes"] or b""
                # 这里只做“收到音频包”的回执，便于你确认 bytes 通了
                await ws.send_json({
                    "code": 0,
                    "info": "got_audio_chunk",
                    "result": {"bytes": len(data)}
                })
                continue

    except WebSocketDisconnect:
        # 前端直接断开会走到这（此时就不能再 send 了）
        print("[WS] client disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
