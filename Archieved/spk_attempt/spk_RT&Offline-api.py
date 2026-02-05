# spk_rt_offline_server.py
# =========================================================
# WebSocket版：实时 VAD→ASR（回传网页） + 断开后离线 spk-cam++ 输出（回传最终JSON）
# 输入：WebSocket 二进制 PCM16LE, 16kHz mono
# 输出：
#   - code=0: 实时分段识别结果（data=text，info带时间戳）
#   - code=1: 断开后离线 spk 输出（data=OUTPUT_JSON的完整内容）
# 同时本地仍会落盘：
#   FULL_WAV_PATH, OUTPUT_JSON, DEBUG_RAW_JSON
# =========================================================
from datetime import datetime
import os
import json
import wave
import time
import asyncio
import tempfile
from typing import List, Dict, Any
from urllib.parse import parse_qs


import numpy as np
import ffmpeg
from funasr import AutoModel

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic import BaseModel
import uvicorn


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


paths = make_output_session("../../output")

FULL_WAV_PATH = paths["wav"]
OUTPUT_JSON = paths["json"]
DEBUG_RAW_JSON = paths["debug"]

# =========================================================
# 音频参数（网页侧必须匹配）
# =========================================================
RATE = 16000
CHANNELS = 1  # mono

# =========================================================
# 实时参数（沿用你 spk-RT&Offline.py）
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
# 离线阶段（完全来自你的 spk-RT&Offline.py）
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


def offline_postprocess(full_wav_path: str):
	print("\n🧾 Offline: VAD → ASR → PUNC → Cam++ (spk-cam++.py style) ...")
	audio_bytes = load_audio_bytes(full_wav_path)
	
	# 你原脚本默认 device="cuda", ngpu=1, ncpu=4，这里保持一致
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


# =========================================================
# FastAPI / WebSocket（参考 server_wss.py 的模式）
# =========================================================
class TranscriptionResponse(BaseModel):
	code: int
	info: str = ""
	data: str = ""


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
#  - 断开时：保存 FULL_WAV_PATH，跑 offline_postprocess，再把最终 JSON 回传（如果还能发）
# =========================================================
@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
	"""
	Query:
	  - lang=auto (default auto)
	Stream:
	  - binary PCM16LE, 16kHz, mono
	"""
	query_params = parse_qs(websocket.scope.get("query_string", b"").decode(errors="ignore"))
	lang = (query_params.get("lang", ["auto"])[0] or "auto").strip()
	
	await websocket.accept()
	print(f"✅ WS connected. lang={lang}")
	
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
	
	# 固定 FULL_WAV_PATH
	ensure_dir_for_file(paths["wav"])
	
	try:
		while True:
			data = await websocket.receive_bytes()
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
						
						# 边界保护
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
	
	except WebSocketDisconnect:
		print("🔌 WS disconnected, start video_processor postprocess...")
	except Exception as e:
		print(f"❌ WS error: {e}")
		try:
			await websocket.close()
		except Exception:
			pass
	finally:
		# 1) 更新完整路径
		FULL_WAV_PATH = paths["wav"]
		OUTPUT_JSON = paths["json"]
		DEBUG_RAW_JSON = paths["debug"]
		
		# 1) 保存完整 wav
		full_i16 = np.concatenate(full_i16_chunks, axis=0) if full_i16_chunks else np.zeros(0, dtype=np.int16)
		save_wav_pcm16(FULL_WAV_PATH, full_i16, RATE)
		print(f"✅ Full audio saved: {FULL_WAV_PATH} (samples={len(full_i16)})")
		
		# 2) 离线 spk 输出（放线程里避免卡 event loop）
		try:
			await asyncio.to_thread(offline_postprocess, FULL_WAV_PATH)
		except Exception as e:
			print(f"❌ offline_postprocess failed: {e}")
			return
		
		# 3) 尝试把最终 JSON 回传（注意：断开后通常无法再发送；如果你前端是“先发final请求再关”，就能收到）
		try:
			if os.path.exists(OUTPUT_JSON):
				with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
					final_obj = json.load(f)
				await websocket.send_json(
					TranscriptionResponse(
						code=1,
						info="final_segments",
						data=json.dumps(final_obj, ensure_ascii=False),
					).model_dump()
				)
		except Exception:
			# 客户端已断开，发送会失败，这里静默即可
			pass

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Run RT(WebSocket) + Offline(speaker clustering) server.")
	parser.add_argument("--host", type=str, default="0.0.0.0")
	parser.add_argument("--port", type=int, default=27000)
	args = parser.parse_args()
	uvicorn.run(app, host=args.host, port=args.port)
