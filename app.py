# app.py
# =========================================================
# WebSocket 实时转写 + 断开后离线说话人合并+ DeepSeek 总结
# =========================================================
# 功能概述：
#  - 接收前端通过 WebSocket 发送的二进制 PCM16LE 音频流（16kHz 单声道），
#    实时做 VAD -> ASR 并将中间分段识别结果回传网页。
#  - 在收到客户端「end」信号或结束后，保存完整 WAV，
#    离线执行 ASR/VAD/PUNC/说话人聚类，生成最终 JSON 并回传。
#  - 离线完成后调用 DeepSeek（本地 Ollama 或云端 API）生成会议纪要 summary 并回传。
#
# 输入（WebSocket）：
#  - 二进制：PCM16LE, 16kHz, mono
#  - 文本：{"type":"end"} 或 纯字符串 "end" 表示录音结束
#
# 输出（WebSocket JSON message）：
#  - code=0: 实时分段识别结果（data=text，info 包含时间戳/asr耗时等）
#  - code=1: 断开后离线说话人合并输出（data=最终 JSON）
#  - code=2: DeepSeek 生成的 summary 文本
#
# 本地落盘（每次连接生成独立 session 目录）：
#  - FULL_WAV_PATH   -> 完整会话 WAV（PCM16）
#  - OUTPUT_JSON     -> 离线合并后最终 segments JSON
#  - DEBUG_RAW_JSON  -> 离线模型原始 sentence_info（调试用）
#
# 关键配置与依赖（环境变量 / 常量）：
#  - 采样率 RATE = 16000, CHANNELS = 1
#  - 实时模型：RT_ASR_MODEL, RT_VAD_MODEL（使用 funasr.AutoModel）
#  - 离线模型路径：通过 get_model_paths() 指向本地 modelscope 缓存
#  - DeepSeek：USE_LOCAL_DEEPSEEK (Ollama 本地) 或 官方云端（DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL）
#  - 需要依赖：python 包 numpy, ffmpeg-python, requests, funasr, fastapi, uvicorn 等；系统需安装 ffmpeg
#
# 运行方式：
#  - 直接运行：python `app.py`，可选参数 --host/--port（默认 0.0.0.0:27000）
#
# 注意事项：
#  - 为避免每次连接重复加载，实时模型在模块级别全局加载。
#  - 离线处理会在后台线程中执行以避免阻塞事件循环。
#  - 该文件同时包含实时 WebSocket 服务与离线后处理逻辑（VAD/ASR/PUNC/spk + DeepSeek）。
# =========================================================

from datetime import datetime
import os
import json
import wave
import time
import asyncio
from typing import List, Dict, Any
from urllib.parse import parse_qs
import argparse

import numpy as np
import ffmpeg
import requests
from funasr import AutoModel

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic import BaseModel
import uvicorn

# 加载本地环境变量（主要是DEEPSEEK_API_KEY）
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# 输出会话目录（按时间戳）
# =========================================================
def make_output_session(base_dir="output"):
	now = datetime.now()
	session_name = now.strftime("%Y%m%d_%H%M%S")
	session_dir = os.path.join(base_dir, session_name)
	
	os.makedirs(session_dir, exist_ok=True)
	
	paths = {
		"base" : session_dir,
		"wav"  : os.path.join(session_dir, "session_full.wav"),
		"json" : os.path.join(session_dir, "stream_output.json"),
		"debug": os.path.join(session_dir, "debug_segments_raw.json"),
	}
	return paths


# 每次 WebSocket 连接（一次录音）都要重新生成 session 目录
paths = None

FULL_WAV_PATH = ""
OUTPUT_JSON = ""
DEBUG_RAW_JSON = ""

# =========================================================
# 音频参数（网页侧必须匹配）
# =========================================================
RATE = 16000
CHANNELS = 1  # mono

# =========================================================
# 实时参数
# =========================================================
RT_VAD_CHUNK_MS = 300
RT_MAX_END_SILENCE_MS = 500

RT_ASR_MODEL = "iic/SenseVoiceSmall"
RT_VAD_MODEL = "fsmn-vad"
RT_VAD_REV = "v2.0.4"

ASR_DEVICE = "cuda:0"
VAD_DEVICE = "cuda:0"

# =========================================================
# 离线输出合并规则（同 spk 且 gap <= MERGE_GAP_MS）
# =========================================================
MERGE_CONTINUOUS_SPK = True
MERGE_GAP_MS = 300

# =========================================================
# DeepSeek 总结配置（支持 Ollama 本地 / 官方云端）
# =========================================================
USE_LOCAL_DEEPSEEK = False  # True = Ollama 本地；False = 官方 DeepSeek API

# ---- Local Ollama DeepSeek ----
OLLAMA_DEEPSEEK_URL = os.getenv(
	"OLLAMA_DEEPSEEK_URL",
	"http://192.168.10.90:11434/api/generate"
)
OLLAMA_DEEPSEEK_MODEL = os.getenv(
	"OLLAMA_DEEPSEEK_MODEL",
	"deepseek-r1:32b"
)

# ---- Cloud DeepSeek ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-xxx":
	raise RuntimeError("DEEPSEEK_API_KEY is not set")


def _deepseek_summarize_ollama(raw_text: str) -> str:
	if not raw_text.strip():
		return ""
	
	prompt = (
			"你是一个会议纪要助手，生成会议纪要。注意：不要使用md格式，只输出纯文本。"
			+ "会议结束的时间为：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			+ raw_text
	)
	
	payload = {
		"model" : OLLAMA_DEEPSEEK_MODEL,
		"prompt": prompt,
		"stream": False,
	}
	
	try:
		resp = requests.post(
			OLLAMA_DEEPSEEK_URL,
			headers={"Content-Type": "application/json"},
			json=payload,
			timeout=120,
		)
		resp.raise_for_status()
		return resp.json().get("response", "").strip()
	except Exception as e:
		print(f"[DeepSeek Ollama Error] {e}")
		return ""


def _deepseek_summarize_cloud(raw_text: str) -> str:
	if not raw_text.strip() or not DEEPSEEK_API_KEY:
		return ""
	
	payload = {
		"model"      : DEEPSEEK_MODEL,
		"messages"   : [
			{"role"   : "system",
			 "content": "你是一个会议纪要助手，生成会议纪要。注意：不要使用md格式，只输出纯文本。" +
			            "会议结束的时间为：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
			{"role": "user", "content": raw_text},
		],
		"temperature": 0.2,
	}
	
	try:
		resp = requests.post(
			f"{DEEPSEEK_BASE_URL}/chat/completions",
			headers={
				"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
				"Content-Type" : "application/json",
			},
			json=payload,
			timeout=60,
		)
		resp.raise_for_status()
		j = resp.json()
		return j["choices"][0]["message"]["content"].strip()
	except Exception as e:
		print(f"[DeepSeek Cloud Error] {e}")
		return ""


def deepseek_summarize(raw_text: str) -> str:
	if USE_LOCAL_DEEPSEEK:
		return _deepseek_summarize_ollama(raw_text)
	else:
		return _deepseek_summarize_cloud(raw_text)


# =========================================================
# 工具：确保目录、WAV 写入（PCM16）
# =========================================================
def ensure_dir_for_file(path: str):
	d = os.path.dirname(path)
	if d:
		os.makedirs(d, exist_ok=True)


def save_wav_pcm16(path: str, audio_i16: np.ndarray, sr: int = 16000):
	ensure_dir_for_file(path)
	with wave.open(path, "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)  # int16
		wf.setframerate(sr)
		wf.writeframes(audio_i16.tobytes())


# =========================================================
# 离线阶段
#   1) ffmpeg 读音频为 wav bytes(16k mono pcm_s16le)
#   2) AutoModel(asr+vad+punc+spk) generate(sentence_timestamp=True)
#   3) sentence_info -> build_output_units(merge gap)
# =========================================================
def get_model_paths() -> Dict[str, str]:
	home = os.path.expanduser("~")
	base = os.path.join(home, ".cache", "modelscope", "hub", "models", "iic")
	return {
		"asr" : os.path.join(base, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
		"vad" : os.path.join(base, "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
		"punc": os.path.join(base, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
		"spk" : os.path.join(base, "speech_campplus_sv_zh-cn_16k-common"),
	}


def load_audio_bytes(audio_path: str) -> bytes:
	audio_bytes, _ = (
		ffmpeg
		.input(audio_path, threads=0)
		.output("-", format="wav", acodec="pcm_s16le", ac=1, ar=16000)
		.run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
	)
	return audio_bytes


def load_offline_model(device: str = "cuda", ngpu: int = 1, ncpu: int = 4) -> AutoModel:
	p = get_model_paths()
	return AutoModel(
		model=p["asr"],
		vad_model=p["vad"],
		punc_model=p["punc"],
		spk_model=p["spk"],
		ngpu=ngpu,
		ncpu=ncpu,
		device=device,
		disable_pbar=True,
		disable_log=True,
		disable_update=True,
	)


def run_offline_asr(model: AutoModel, audio_bytes: bytes) -> Dict[str, Any]:
	res = model.generate(
		input=audio_bytes,
		batch_size_s=300,
		is_final=True,
		sentence_timestamp=True,
	)
	return res[0] if res else {}


def build_output_units(sentence_info: List[Dict[str, Any]],
                       merge_continuous_spk: bool,
                       merge_gap_ms: int) -> List[Dict[str, Any]]:
	units: List[Dict[str, Any]] = []
	for s in sentence_info or []:
		text = (s.get("text") or "").strip()
		if not text:
			continue
		
		spk = s.get("spk")
		start = int(s.get("start", 0))
		end = int(s.get("end", 0))
		
		if merge_continuous_spk and units:
			last = units[-1]
			same_spk = (last["spk"] == spk)
			gap = start - last["end_ms"]
			if same_spk and gap <= merge_gap_ms:
				last["text"] += text
				last["end_ms"] = end
				continue
		
		units.append({
			"spk"     : spk,
			"text"    : text,
			"start_ms": start,
			"end_ms"  : end,
		})
	return units


def save_json(obj: Dict[str, Any], path: str):
	ensure_dir_for_file(path)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def offline_postprocess(full_wav_path: str) -> Dict[str, Any]:
	"""
	跑离线：VAD → ASR → PUNC → Cam++ (spk) ，落盘 JSON，并返回 out_obj
	"""
	print("\n🧾 Offline: VAD → ASR → PUNC → Cam++ ...")
	audio_bytes = load_audio_bytes(full_wav_path)
	
	model = load_offline_model(device="cuda", ngpu=1, ncpu=4)
	rec = run_offline_asr(model, audio_bytes)
	
	# 原始 sentence_info（debug）
	debug_obj = {
		"audio_full"   : {"path": FULL_WAV_PATH, "sample_rate": RATE, "channels": 1},
		"sentence_info": rec.get("sentence_info", []),
	}
	save_json(debug_obj, DEBUG_RAW_JSON)
	
	# 合并后的 segments（最终输出）
	units = build_output_units(
		rec.get("sentence_info", []),
		MERGE_CONTINUOUS_SPK,
		MERGE_GAP_MS,
	)
	
	out_obj = {
		"audio_full": {"path": FULL_WAV_PATH, "sample_rate": RATE, "channels": 1},
		"segments"  : units,
	}
	save_json(out_obj, OUTPUT_JSON)
	
	print(f"✅ Saved:\n  - {FULL_WAV_PATH}\n  - {OUTPUT_JSON}\n  - {DEBUG_RAW_JSON}")
	return out_obj


def build_transcript_for_summary(result_obj: Dict[str, Any]) -> str:
	"""
	把离线 segments 拼成适合总结的全文（带 speaker/时间戳）
	"""
	segs = result_obj.get("segments") or []
	lines = []
	for s in segs:
		spk = s.get("spk", "spk")
		t0 = int(s.get("start_ms", 0))
		t1 = int(s.get("end_ms", 0))
		txt = (s.get("text") or "").strip()
		if not txt:
			continue
		lines.append(f"[{t0}-{t1}] {spk}: {txt}")
	return "\n".join(lines).strip()


# =========================================================
# FastAPI / WebSocket
# =========================================================
# code 语义定义：
# code = 0  【实时识别分段结果 / partial segment】
#   - 触发时机：
#       每一次 VAD 结束 + ASR 产出文本
#   - info:
#       JSON 字符串，包含时间与耗时信息
#   - data:
#       该分段对应的识别文本
# -------------------------
# code = 1  【最终字幕识别结果 / final result】
#   - 触发时机：
#       WebSocket 即将结束前，整段音频离线处理完成
#   - info:
#       固定字符串："final"
#   - data:
#       final_obj 的 JSON 字符串（ensure_ascii=False）
#       结构示例：
#       {
#         "audio_full": {...},
#         "segments": [
#           {
#             "spk": <speaker_id>,
#             "text": "...",
#             "start_ms": ...,
#             "end_ms": ...
#           }
#         ]
#       }
#
# -------------------------
# code = 2  【状态 / 事件通知（非 ASR 文本）】
#   - 触发时机：
#       WebSocket 结束并回传最后的ai总结
#   - info:
#       固定字符串："summary"
#   data:
#     - summary           : AI总结的summary 文本
#
# =========================================================
class TranscriptionResponse(BaseModel):
	code: int  # 消息类型 / 状态码
	info: str = ""  # 元信息（JSON 字符串 / 标记字符串）
	data: str = ""  # 实际载荷（文本 / JSON 字符串）


app = FastAPI()
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
	if isinstance(exc, HTTPException):
		status_code = exc.status_code
		message = str(exc.detail)
	elif isinstance(exc, RequestValidationError):
		status_code = HTTP_422_UNPROCESSABLE_ENTITY
		message = "Validation error: " + str(exc.errors())
	else:
		status_code = 500
		message = "Internal server error: " + str(exc)
	return JSONResponse(
		status_code=status_code,
		content=TranscriptionResponse(code=status_code, info=message, data="").model_dump(),
	)


@app.get("/health")
async def health():
	return {"ok": True}


# =========================================================
# 全局加载实时模型（避免每个WS连接都加载一次）
# =========================================================
rt_vad = AutoModel(
	model=RT_VAD_MODEL,
	model_revision=RT_VAD_REV,
	disable_pbar=True,
	max_end_silence_time=RT_MAX_END_SILENCE_MS,
	device=VAD_DEVICE,
	disable_update=True,
)

rt_asr = AutoModel(
	model=RT_ASR_MODEL,
	trust_remote_code=True,
	device=ASR_DEVICE,
	disable_update=True,
)


# =========================================================
# WebSocket: /ws/transcribe
#  - 收到音频流：实时 VAD→ASR 回传
#  - 断开时：保存 FULL_WAV_PATH，跑 offline_postprocess，再把最终 JSON + deepseek summary 回传（如果还能发）
# =========================================================
@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
	"""
	Query:
	  - lang=auto (default auto)
	  - sv=0/1 (目前流程A里先不影响逻辑，你后面要用再扩展)
	Stream:
	  - binary PCM16LE, 16kHz, mono
	  - text/json: {"type":"end"} 或 "end" 表示录音结束（流程A关键）
	"""
	# 每次连接都创建新的会话目录（一次录音一次目录）
	local_paths = make_output_session("output")
	local_full_wav = local_paths["wav"]
	local_output_json = local_paths["json"]
	local_debug_json = local_paths["debug"]
	
	query_params = parse_qs(websocket.scope.get("query_string", b"").decode(errors="ignore"))
	lang = (query_params.get("lang", ["auto"])[0] or "auto").strip()
	
	await websocket.accept()
	print(f"✅ WS connected. lang={lang} session={local_paths['base']}")
	
	chunk_size = int(RT_VAD_CHUNK_MS * RATE / 1000)
	
	# bytes 对齐缓冲
	buf_bytes = b""
	
	# float32 缓冲
	audio_buffer = np.array([], dtype=np.float32)
	audio_vad = np.array([], dtype=np.float32)
	
	# 保存全量 int16（用于最终离线）
	full_i16_chunks: List[np.ndarray] = []
	
	cache_vad: Dict[str, Any] = {}
	cache_asr: Dict[str, Any] = {}
	
	last_vad_beg = -1
	last_vad_end = -1
	offset_ms = 0
	
	ensure_dir_for_file(local_full_wav)
	
	# 流程A：通过“end”信号结束 while 循环，而不是靠断开
	ended_by_client = False
	
	try:
		while True:
			msg = await websocket.receive()
			
			# 1) 前端发送二进制音频
			if "bytes" in msg and msg["bytes"] is not None:
				data = msg["bytes"]
				if not data:
					continue
				
				buf_bytes += data
				if len(buf_bytes) < 2:
					continue
				
				aligned_len = len(buf_bytes) - (len(buf_bytes) % 2)
				if aligned_len <= 0:
					continue
				
				audio_i16 = np.frombuffer(buf_bytes[:aligned_len], dtype=np.int16)
				buf_bytes = buf_bytes[aligned_len:]
				
				if audio_i16.size == 0:
					continue
				
				full_i16_chunks.append(audio_i16.copy())
				
				audio_f32 = audio_i16.astype(np.float32) / 32767.0
				audio_buffer = np.append(audio_buffer, audio_f32)
				
				while len(audio_buffer) >= chunk_size:
					chunk = audio_buffer[:chunk_size]
					audio_buffer = audio_buffer[chunk_size:]
					audio_vad = np.append(audio_vad, chunk)
					
					res_vad = rt_vad.generate(
						input=chunk,
						cache=cache_vad,
						is_final=False,
						chunk_size=RT_VAD_CHUNK_MS,
					)
					
					values = []
					if res_vad and isinstance(res_vad, list) and res_vad[0].get("value") is not None:
						values = res_vad[0]["value"]
					if not values:
						continue
					
					for seg in values:
						if seg[0] > -1:
							last_vad_beg = seg[0]
						if seg[1] > -1:
							last_vad_end = seg[1]
						
						if last_vad_beg > -1 and last_vad_end > -1:
							beg_ms_local = last_vad_beg - offset_ms
							end_ms_local = last_vad_end - offset_ms
							offset_ms += end_ms_local
							
							beg = int(beg_ms_local * RATE / 1000)
							end = int(end_ms_local * RATE / 1000)
							
							beg = max(beg, 0)
							end = min(end, len(audio_vad))
							
							speech = audio_vad[beg:end]
							
							t0 = time.time()
							asr_out = rt_asr.generate(
								input=speech,
								cache=cache_asr,
								language=lang,
								use_itn=True,
							)
							t1 = time.time()
							
							text = ""
							if asr_out and isinstance(asr_out, list):
								text = (asr_out[0].get("text") or "").strip()
							
							seg_end_ms_global = int(offset_ms)
							seg_start_ms_global = int(offset_ms - end_ms_local)
							
							if text:
								await websocket.send_json(
									TranscriptionResponse(
										code=0,
										info=json.dumps(
											{
												"start_ms": seg_start_ms_global,
												"end_ms"  : seg_end_ms_global,
												"asr_ms"  : int((t1 - t0) * 1000),
											},
											ensure_ascii=False,
										),
										data=text,
									).model_dump()
								)
							
							# 消费已用音频
							audio_vad = audio_vad[end:]
							last_vad_beg = -1
							last_vad_end = -1
				
				continue
			
			# 2) 前端发送文本（结束信号）
			if "text" in msg and msg["text"] is not None:
				txt = (msg["text"] or "").strip()
				if not txt:
					continue
				
				# 支持两种：纯字符串 "end" / JSON {"type":"end"}
				is_end = False
				if txt.lower() in ("end", "stop", "finish", "done"):
					is_end = True
				else:
					try:
						j = json.loads(txt)
						if isinstance(j, dict) and str(j.get("type", "")).lower() in ("end", "stop", "finish", "done"):
							is_end = True
					except Exception:
						pass
				
				if is_end:
					print("🟦 Received end signal from client. Start video_processor postprocess...")
					ended_by_client = True
					break
	
	except WebSocketDisconnect:
		print("🔌 WS disconnected (client closed before end).")
	except Exception as e:
		print(f"❌ WS error: {e}")
		try:
			await websocket.close()
		except Exception:
			pass
		return
	
	# =========================
	# 这里开始“录音结束后的离线流程”，但 WS 仍然保持打开
	# =========================
	try:
		# 1) 保存完整 wav
		full_i16 = np.concatenate(full_i16_chunks, axis=0) if full_i16_chunks else np.zeros(0, dtype=np.int16)
		save_wav_pcm16(local_full_wav, full_i16, RATE)
		print(f"✅ Full audio saved: {local_full_wav} (samples={len(full_i16)})")
		
		# 2) 离线 spk 输出（线程避免卡 event loop）
		def _offline_postprocess_local(full_wav_path: str) -> Dict[str, Any]:
			# 临时覆盖全局输出路径，让 offline_postprocess 落到本次 session 目录
			global FULL_WAV_PATH, OUTPUT_JSON, DEBUG_RAW_JSON
			FULL_WAV_PATH = local_full_wav
			OUTPUT_JSON = local_output_json
			DEBUG_RAW_JSON = local_debug_json
			return offline_postprocess(full_wav_path)
		
		final_obj: Dict[str, Any] = await asyncio.to_thread(_offline_postprocess_local, local_full_wav)
		
		# 3) DeepSeek 总结
		try:
			transcript = build_transcript_for_summary(final_obj)
			print("这是用于 DeepSeek 总结的 transcript 内容预览：")
			print("-----")
			print(transcript)
			print("-----")
			summary_text = await asyncio.to_thread(deepseek_summarize, transcript)
			print("✅ DeepSeek summary generated.")
		except Exception as e:
			print(f"❌ deepseek summarize failed: {e}")
			summary_text = ""
		# summary_text = "这是测试用的summary占位符"
		
		# 4) 回传最终 JSON (code=1)
		try:
			await websocket.send_json(
				TranscriptionResponse(
					code=1,
					info="final",
					data=json.dumps(final_obj, ensure_ascii=False),
				).model_dump()
			)
		except Exception as e:
			print(f"⚠️ send final failed (maybe client already closed): {e}")
		
		# 5) 回传 summary (code=2)
		try:
			await websocket.send_json(
				TranscriptionResponse(
					code=2,
					info="summary",
					data=summary_text,
				).model_dump()
			)
		except Exception as e:
			print(f"⚠️ send final failed (maybe client already closed): {e}")
	
	finally:
		# 服务端主动关闭（前端也会在收到 code=1 后 close）
		try:
			await websocket.close()
		except Exception:
			pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Run RT(WebSocket) + Offline(speaker clustering) + DeepSeek summary server.")
	parser.add_argument("--host", type=str, default="0.0.0.0")
	parser.add_argument("--port", type=int, default=27000)
	args = parser.parse_args()
	uvicorn.run(app, host=args.host, port=args.port)
