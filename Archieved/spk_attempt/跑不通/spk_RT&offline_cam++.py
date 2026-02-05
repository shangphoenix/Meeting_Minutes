# spk_RT&offline_cam++.py
# 这个的问题是:
# 结束之后没有再跑一次ASR+VAD,导致CAM++无法识别不同说话人
import json
import os
import re
import time
import math
import wave
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
import pyaudio
import torch
from funasr import AutoModel

# =========================
# 0) 参数
# =========================
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024

# VAD 流式 chunk 大小（ms）
CHUNK_MS = 300
MAX_END_SILENCE_MS = 500

# 离线声纹前，先把碎段合并成长块
SPK_PREMERGE_GAP_MS = 800  # 两段之间 <= 800ms 就先拼起来算声纹
SPK_MIN_CHUNK_MS = 2500  # 拼出来的块长度 < 2.5s 的声纹很不稳（可调）
SPK_MIN_EMB_MS = 1200  # 最低 1.2s 才尝试算 emb（兜底）

# 最终输出再按同 spk 合并
FINAL_MERGE_GAP_MS = 300  # 同 spk 且 gap <= 300ms，拼文本
FINAL_MERGE_CONTINUOUS_SPK = True

# 说话人在线聚类阈值（越大越不容易把不同人合并）
SPK_COS_THRESH = 0.80

# 输出
OUTPUT_JSON_PATH = "output/realtime_captions.json"
OUTPUT_AUDIO_PATH = "output/realtime_captions.wav"

# 设备（按需改）
ASR_DEVICE = "cuda:0"
VAD_DEVICE = "cuda:0"
SPK_DEVICE = "cuda:0"

# =========================
# 1) 实时模型（只加载 ASR + VAD）
# =========================
model_asr = AutoModel(
	model="iic/SenseVoiceSmall",
	trust_remote_code=True,
	device=ASR_DEVICE,
	disable_update=True
)

model_vad = AutoModel(
	model="fsmn-vad",
	model_revision="v2.0.4",
	disable_pbar=True,
	max_end_silence_time=MAX_END_SILENCE_MS,
	disable_update=True
)

# =========================
# 2) 离线声纹：Cam++ 模型（停止后才加载/调用）
# =========================
SPK_MODEL_NAME = "iic/speech_campplus_sv_zh-cn_16k-common"


# =========================
# 3) Online clustering：assign_spk
# =========================
def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	n = float(np.linalg.norm(x)) + eps
	return x / n


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
	a = _l2norm(a)
	b = _l2norm(b)
	return float(np.dot(a, b))


@dataclass
class SpeakerAssigner:
	cos_thresh: float = SPK_COS_THRESH
	centroids: List[np.ndarray] = field(default_factory=list)
	counts: List[int] = field(default_factory=list)
	
	def assign_spk(self, emb: np.ndarray) -> int:
		if emb is None:
			return -1
		emb = np.asarray(emb, dtype=np.float32).reshape(-1)
		emb = _l2norm(emb)
		
		if not self.centroids:
			self.centroids.append(emb)
			self.counts.append(1)
			return 0
		
		sims = [cosine_sim(emb, c) for c in self.centroids]
		best_i = int(np.argmax(sims))
		best_s = sims[best_i]
		
		if best_s >= self.cos_thresh:
			k = self.counts[best_i]
			new_c = (self.centroids[best_i] * k + emb) / (k + 1.0)
			self.centroids[best_i] = _l2norm(new_c)
			self.counts[best_i] = k + 1
			return best_i
		
		self.centroids.append(emb)
		self.counts.append(1)
		return len(self.centroids) - 1


# =========================
# 4) Cam++ embedding 提取（支持 CUDA tensor）
# =========================
def _to_numpy(x) -> Optional[np.ndarray]:
	if x is None:
		return None
	if torch.is_tensor(x):
		return x.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
	arr = np.asarray(x, dtype=np.float32)
	return arr.reshape(-1)


def extract_spk_embedding(model: AutoModel, speech_f32: np.ndarray) -> Optional[np.ndarray]:
	"""
	尽量兼容不同 funasr/campplus 输出结构，并安全转 numpy
	"""
	try:
		out = model.generate(input=speech_f32)
		if not out:
			return None
		obj = out[0] if isinstance(out, list) else out
		
		candidate_keys = [
			"spk_embedding", "speaker_embedding", "embedding", "emb",
			"spk_emb", "xvector", "vector"
		]
		
		# 1) 直接 tensor/list/ndarray
		if torch.is_tensor(obj) or isinstance(obj, (list, tuple, np.ndarray)):
			return _to_numpy(obj)
		
		# 2) dict
		if isinstance(obj, dict):
			for k in candidate_keys:
				if k in obj and obj[k] is not None:
					return _to_numpy(obj[k])
			
			for nest_k in ["value", "outputs", "result", "data"]:
				v = obj.get(nest_k)
				if isinstance(v, dict):
					for k in candidate_keys:
						if k in v and v[k] is not None:
							return _to_numpy(v[k])
				elif v is not None and (torch.is_tensor(v) or isinstance(v, (list, tuple, np.ndarray))):
					arr = _to_numpy(v)
					if arr is not None and 8 <= arr.size <= 4096:
						return arr
			
			# 兜底：遍历 values 找“像 embedding”的向量
			for v in obj.values():
				if v is None:
					continue
				if torch.is_tensor(v) or isinstance(v, (list, tuple, np.ndarray)):
					arr = _to_numpy(v)
					if arr is not None and arr.ndim == 1 and 8 <= arr.size <= 4096:
						return arr
		
		return None
	except Exception as e:
		print(f"[WARN] extract_spk_embedding failed: {e}")
		return None


# =========================
# 5) 合并工具：先拼长块用于声纹，再最终按 spk 合并输出
# =========================
def remove_tags(text: str) -> str:
	"""
	简单去标签（如 <noise>、[laughter] 等）
	"""
	return re.sub(r"[\[<][^]>]*[>\]]", "", text).strip()


def save_wav(path: str, audio_f32: np.ndarray, sr: int = 16000):
	audio_i16 = np.clip(audio_f32, -1.0, 1.0)
	audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
	
	os.makedirs(os.path.dirname(path), exist_ok=True)
	
	with wave.open(path, "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)
		wf.setframerate(sr)
		wf.writeframes(audio_i16.tobytes())


def merge_for_speaker(segments_raw: List[Dict[str, Any]],
                      gap_ms: int = SPK_PREMERGE_GAP_MS) -> List[Dict[str, Any]]:
	"""
	把实时碎段先拼成更长的“声纹块”，Cam++ 更稳定
	"""
	merged = []
	cur = None
	
	for s in segments_raw:
		if cur is None:
			cur = {
				"start_ms": int(s["start_ms"]),
				"end_ms"  : int(s["end_ms"]),
				"text"    : (s.get("text") or "").strip(),
				"audio"   : s["audio"].astype(np.float32),
				"children": [s],  # 保留映射关系（可选）
			}
			continue
		
		gap = int(s["start_ms"]) - int(cur["end_ms"])
		if gap <= gap_ms:
			cur["end_ms"] = int(s["end_ms"])
			cur["text"] = (cur["text"] + (s.get("text") or "")).strip()
			cur["audio"] = np.concatenate([cur["audio"], s["audio"].astype(np.float32)], axis=0)
			cur["children"].append(s)
		else:
			merged.append(cur)
			cur = {
				"start_ms": int(s["start_ms"]),
				"end_ms"  : int(s["end_ms"]),
				"text"    : (s.get("text") or "").strip(),
				"audio"   : s["audio"].astype(np.float32),
				"children": [s],
			}
	
	if cur is not None:
		merged.append(cur)
	
	return merged


def final_merge_by_spk(chunks_with_spk: List[Dict[str, Any]],
                       gap_ms: int = FINAL_MERGE_GAP_MS) -> List[Dict[str, Any]]:
	"""
	最终输出合并：同 spk 且 gap 小就拼文本、扩 end
	"""
	out = []
	for c in chunks_with_spk:
		text = (c.get("text") or "").strip()
		if not text:
			continue
		spk = int(c.get("spk", -1))
		start_ms = int(c["start_ms"])
		end_ms = int(c["end_ms"])
		
		if FINAL_MERGE_CONTINUOUS_SPK and out:
			last = out[-1]
			if last["spk"] == spk and (start_ms - last["end_ms"]) <= gap_ms:
				last["text"] += text
				last["end_ms"] = end_ms
				continue
		
		out.append({
			"spk"     : spk,
			"text"    : text,
			"start_ms": start_ms,
			"end_ms"  : end_ms
		})
	return out


# =========================
# 6) 主流程：实时 VAD+ASR 录制片段；停止后离线 Cam++
# =========================
def main():
	# Full audio buffer (for saving)
	full_audio = np.array([], dtype=np.float32)
	
	# PyAudio
	p = pyaudio.PyAudio()
	stream = p.open(
		format=FORMAT,
		channels=CHANNELS,
		rate=RATE,
		input=True,
		frames_per_buffer=CHUNK,
	)
	print("🎙️ Start streaming (Ctrl+C to stop). [Scheme A: NO real-time Cam++]")
	
	chunk_size = int(CHUNK_MS * RATE / 1000)
	
	audio_buffer = np.array([], dtype=np.float32)  # 原始采集缓存
	audio_vad = np.array([], dtype=np.float32)  # VAD 时间轴缓存
	
	cache_vad: Dict[str, Any] = {}
	cache_asr: Dict[str, Any] = {}
	
	last_vad_beg = -1
	last_vad_end = -1
	offset_ms = 0  # 用于把 VAD 输出映射成当前 audio_vad 局部坐标（沿用你实时脚本的思路）
	
	segments_raw: List[Dict[str, Any]] = []
	
	try:
		while True:
			data = stream.read(CHUNK, exception_on_overflow=False)
			audio_i16 = np.frombuffer(data, dtype=np.int16)
			audio_f32 = audio_i16.astype(np.float32) / 32767.0
			audio_buffer = np.append(audio_buffer, audio_f32)
			full_audio = np.append(full_audio, audio_f32)
			
			while len(audio_buffer) >= chunk_size:
				chunk = audio_buffer[:chunk_size]
				audio_buffer = audio_buffer[chunk_size:]
				audio_vad = np.append(audio_vad, chunk)
				
				# ===== VAD streaming =====
				res_vad = model_vad.generate(
					input=chunk,
					cache=cache_vad,
					is_final=False,
					chunk_size=CHUNK_MS
				)
				
				values = []
				if res_vad and isinstance(res_vad, list) and res_vad[0].get("value") is not None:
					values = res_vad[0]["value"]
				
				if values:
					for seg in values:
						if seg[0] > -1:
							last_vad_beg = seg[0]
						if seg[1] > -1:
							last_vad_end = seg[1]
						
						if last_vad_beg > -1 and last_vad_end > -1:
							# seg 是 ms（相对累计时间轴），映射到当前 audio_vad
							beg_ms_local = last_vad_beg - offset_ms
							end_ms_local = last_vad_end - offset_ms
							
							# 消费到 end：更新 offset_ms
							offset_ms += end_ms_local
							
							beg = int(beg_ms_local * RATE / 1000)
							end = int(end_ms_local * RATE / 1000)
							speech = audio_vad[beg:end]
							
							# ===== ASR =====
							asr_out = model_asr.generate(
								input=speech,
								cache=cache_asr,
								language="auto",
								use_itn=True
							)
							text = ""
							if asr_out and isinstance(asr_out, list):
								text = (asr_out[0].get("text") or "").strip()
							
							# 全局时间（ms）
							seg_end_ms_global = int(offset_ms)
							seg_start_ms_global = int(offset_ms - end_ms_local)
							
							segments_raw.append({
								"start_ms": seg_start_ms_global,
								"end_ms"  : seg_end_ms_global,
								"text"    : text,
								"audio"   : speech.astype(np.float32),
							})
							
							# 实时打印
							if text:
								print(
									f"[{seg_start_ms_global / 1000:07.3f}-{seg_end_ms_global / 1000:07.3f}] {remove_tags(text)}")
								print(
									f"[{seg_start_ms_global / 1000:07.3f}-{seg_end_ms_global / 1000:07.3f}] {text}")
							
							# 清理已消费音频
							audio_vad = audio_vad[end:]
							last_vad_beg = -1
							last_vad_end = -1
	
	except KeyboardInterrupt:
		print("\n⏹️ stopped. Start video_processor Cam++ speaker assignment...")
	
	finally:
		stream.stop_stream()
		stream.close()
		p.terminate()
		save_wav(OUTPUT_AUDIO_PATH, full_audio, sr=RATE)
		print(f"✅ saved full audio: {OUTPUT_AUDIO_PATH}")
	
	# =========================
	# 7) 离线阶段：拼长块 -> Cam++ -> 聚类 -> 最终合并输出
	# =========================
	if not segments_raw:
		with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
			json.dump({"segments": []}, f, ensure_ascii=False, indent=2)
		print(f"✅ saved empty: {OUTPUT_JSON_PATH}")
		return
	
	# 7.1 先拼长块（降低声纹误差）
	spk_chunks = merge_for_speaker(segments_raw, gap_ms=SPK_PREMERGE_GAP_MS)
	
	# 7.2 加载 Cam++（只在离线阶段）
	model_spk = AutoModel(
		model=SPK_MODEL_NAME,
		device=SPK_DEVICE,
		disable_update=True
	)
	
	assigner = SpeakerAssigner(cos_thresh=SPK_COS_THRESH)
	
	# 7.3 逐块算 embedding 并分配 spk
	for c in spk_chunks:
		dur_ms = int(len(c["audio"]) * 1000 / RATE)
		spk_id = -1
		
		# 太短的块：不给 spk（或你也可以选择并到相邻块）
		if dur_ms >= SPK_MIN_EMB_MS and dur_ms >= SPK_MIN_CHUNK_MS:
			emb = extract_spk_embedding(model_spk, c["audio"])
			if emb is not None:
				spk_id = assigner.assign_spk(emb)
		
		c["spk"] = spk_id
	
	# 7.4 最终输出再按同 spk 合并（更像你离线脚本的结果）
	final_segments = final_merge_by_spk(spk_chunks, gap_ms=FINAL_MERGE_GAP_MS)
	
	with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
		json.dump({"segments": final_segments}, f, ensure_ascii=False, indent=2)
	
	print(
		f"✅ saved: {OUTPUT_JSON_PATH} (raw={len(segments_raw)}, spk_chunks={len(spk_chunks)}, final={len(final_segments)})")


if __name__ == "__main__":
	main()
