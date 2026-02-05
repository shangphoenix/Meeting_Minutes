# spk_RT_cam++.py
# 这个方案是实时流式采集音频，然后用 VAD 切段，切好的段再送 ASR + SPK 模型推理，
# 最后在线合并输出单元并打印，同时保存到 JSON 文件。
# 说话人聚类用的是 Cam++ / CAMPPlus 模型提取的
# 但是有一个较大的问题:
# 由于 Cam++ / CAMPPlus 模型本身并不是为实时设计的，
# 实时识别会导致说话人分配不稳定，容易出现同一说话人被分配成多个 ID 的情况。

import json
import time
import numpy as np
import pyaudio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import torch
from funasr import AutoModel

# =========================
# 0) 参数
# =========================
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024  # PyAudio 每次 read 的帧数
CHUNK_MS = 300  # VAD 流式 chunk 大小（ms）
MAX_END_SILENCE_MS = 500  # 你原实时脚本里用的

# 合并规则（来自离线脚本）
MERGE_CONTINUOUS_SPK = True
MERGE_GAP_MS = 300

# 说话人在线聚类阈值（经验值：越大越“保守”更容易新建说话人）
SPK_COS_THRESH = 0.65
SPK_MIN_SPEECH_MS = 800  # 太短的段不算声纹（避免抖动）

OUTPUT_JSON = "output/stream_output.json"

# =========================
# 1) 模型加载
# =========================
# ASR：沿用你的实时方案
model_asr = AutoModel(
	model="iic/SenseVoiceSmall",
	trust_remote_code=True,
	device="cuda:0",
	disable_update=True
)

# VAD：沿用你的实时方案
model_vad = AutoModel(
	model="fsmn-vad",
	model_revision="v2.0.4",
	disable_pbar=True,
	max_end_silence_time=MAX_END_SILENCE_MS,
	disable_update=True
)

# SPK：Cam++ / CAMPPlus（离线方案里的 spk_model）
# 注意：不同版本 funasr 的输出 key 可能不一致，所以下面有 robust 解析
model_spk = AutoModel(
	model="iic/speech_campplus_sv_zh-cn_16k-common",
	device="cuda:0",
	disable_update=True
)


# =========================
# 2) 在线说话人分配器
# =========================
def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	n = np.linalg.norm(x) + eps
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
		"""给一个 embedding，返回 spk_id（0,1,2,...），并在线更新 centroid。"""
		if emb is None:
			return -1
		emb = np.asarray(emb, dtype=np.float32).reshape(-1)
		
		if len(self.centroids) == 0:
			self.centroids.append(_l2norm(emb))
			self.counts.append(1)
			return 0
		
		sims = [cosine_sim(emb, c) for c in self.centroids]
		best_i = int(np.argmax(sims))
		best_s = sims[best_i]
		
		if best_s >= self.cos_thresh:
			# 在线更新 centroid：加权平均再归一化
			k = self.counts[best_i]
			new_c = (self.centroids[best_i] * k + _l2norm(emb)) / (k + 1.0)
			self.centroids[best_i] = _l2norm(new_c)
			self.counts[best_i] = k + 1
			return best_i
		
		# 新 speaker
		self.centroids.append(_l2norm(emb))
		self.counts.append(1)
		return len(self.centroids) - 1


def _to_numpy(x):
	"""把各种类型的 embedding 转成 1D numpy.float32"""
	if x is None:
		return None
	# torch tensor
	if torch.is_tensor(x):
		return x.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
	# numpy / list
	arr = np.asarray(x, dtype=np.float32)
	return arr.reshape(-1)


def extract_spk_embedding(model, speech_f32: np.ndarray):
	"""
	从 Cam++ / CAMPPlus 输出中提取 embedding，并安全转 numpy（支持 cuda tensor）
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
		
		# 1) 直接就是 tensor / list / np
		if torch.is_tensor(obj) or isinstance(obj, (list, tuple, np.ndarray)):
			return _to_numpy(obj)
		
		# 2) dict：常见结构
		if isinstance(obj, dict):
			for k in candidate_keys:
				if k in obj and obj[k] is not None:
					return _to_numpy(obj[k])
			
			# 嵌套 dict：value / outputs / result / data
			for nest_k in ["value", "outputs", "result", "data"]:
				v = obj.get(nest_k)
				if isinstance(v, dict):
					for k in candidate_keys:
						if k in v and v[k] is not None:
							return _to_numpy(v[k])
				# 有的 nest 直接就是 tensor
				if v is not None and (torch.is_tensor(v) or isinstance(v, (list, tuple, np.ndarray))):
					# 但只有当它像 embedding（1D/2D小向量）时才收
					arr = _to_numpy(v)
					if arr is not None and arr.size <= 4096:
						return arr
		
		# 3) 实在不行：遍历 dict 里的所有值，找到第一个“像 embedding”的向量
		if isinstance(obj, dict):
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
# 3) 合并输出单元（来自离线逻辑）
# =========================
def merge_units(units: List[Dict[str, Any]], spk: int, text: str, start_ms: int, end_ms: int):
	text = (text or "").strip()
	if not text:
		return
	
	if MERGE_CONTINUOUS_SPK and units:
		last = units[-1]
		same_spk = (last["spk"] == spk)
		gap = start_ms - last["end_ms"]
		if same_spk and gap <= MERGE_GAP_MS:
			last["text"] += text
			last["end_ms"] = end_ms
			return
	
	units.append({
		"spk"     : spk,
		"text"    : text,
		"start_ms": int(start_ms),
		"end_ms"  : int(end_ms),
	})


# =========================
# 4) 主流程：采集 -> VAD -> 切段 -> ASR + SPK -> merge -> JSON
# =========================
def main():
	# PyAudio
	p = pyaudio.PyAudio()
	stream = p.open(
		format=FORMAT,
		channels=CHANNELS,
		rate=RATE,
		input=True,
		frames_per_buffer=CHUNK,
	)
	print("🎙️ Streaming start (Ctrl+C to stop)")
	
	# 状态
	chunk_size = int(CHUNK_MS * RATE / 1000)
	
	audio_buffer = np.array([], dtype=np.float32)  # 原始采集缓存
	audio_vad = np.array([], dtype=np.float32)  # VAD 参考时间轴缓存
	
	cache_vad: Dict[str, Any] = {}
	cache_asr: Dict[str, Any] = {}
	
	last_vad_beg = -1
	last_vad_end = -1
	offset = 0  # 你原脚本里的 offset 逻辑：把 VAD 输出映射到 audio_vad 子数组
	
	# 结果
	units: List[Dict[str, Any]] = []
	spk_assigner = SpeakerAssigner()
	
	# 记录“从开始到现在”的毫秒
	t0 = time.time()
	
	try:
		while True:
			data = stream.read(CHUNK, exception_on_overflow=False)
			audio_i16 = np.frombuffer(data, dtype=np.int16)
			audio_f32 = audio_i16.astype(np.float32) / 32767.0
			audio_buffer = np.append(audio_buffer, audio_f32)
			
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
				
				# res_vad[0]["value"] 可能是 [] / [[beg,-1]] / [[-1,end]] / [[beg,end],...]
				if res_vad and len(res_vad[0].get("value", [])) > 0:
					for seg in res_vad[0]["value"]:
						if seg[0] > -1:
							last_vad_beg = seg[0]
						if seg[1] > -1:
							last_vad_end = seg[1]
						
						# 只有 beg/end 都拿到，才切段
						if last_vad_beg > -1 and last_vad_end > -1:
							# 映射到 audio_vad 的局部坐标（沿用你原实时脚本）
							last_vad_beg -= offset
							last_vad_end -= offset
							offset += last_vad_end
							
							beg = int(last_vad_beg * RATE / 1000)
							end = int(last_vad_end * RATE / 1000)
							speech = audio_vad[beg:end]
							
							seg_dur_ms = int((end - beg) * 1000 / RATE)
							# ===== ASR =====
							asr_out = model_asr.generate(
								input=speech,
								cache=cache_asr,
								language="auto",
								use_itn=True,
							)
							text = ""
							if asr_out and asr_out[0].get("text"):
								text = asr_out[0]["text"]
							
							# ===== SPK =====
							spk_id = -1
							if seg_dur_ms >= SPK_MIN_SPEECH_MS:
								emb = extract_spk_embedding(model_spk, speech)
								if emb is not None:
									spk_id = spk_assigner.assign_spk(emb)
							
							# 估计该段在全局时间轴的起止（ms）
							# 这里用“从开始到现在”的近似：offset 其实就是 end（ms）在 audio_vad 的累计，
							# 所以用 (offset - last_vad_end .. offset) 作为该段的全局时间（ms）
							seg_end_ms_global = offset
							seg_start_ms_global = offset - int(last_vad_end)
							
							# merge + 打印
							if text.strip():
								merge_units(units, spk_id, text, seg_start_ms_global, seg_end_ms_global)
								print(
									f"[{seg_start_ms_global / 1000:07.3f}-{seg_end_ms_global / 1000:07.3f}] [SPK{spk_id}] {text}")
							
							# 清理已消费音频
							audio_vad = audio_vad[end:]
							last_vad_beg = -1
							last_vad_end = -1
	
	except KeyboardInterrupt:
		print("\n⏹️ stopped")
	
	finally:
		stream.stop_stream()
		stream.close()
		p.terminate()
		
		with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
			json.dump({"segments": units}, f, ensure_ascii=False, indent=2)
		
		print(f"✅ saved: {OUTPUT_JSON} (segments={len(units)})")


if __name__ == "__main__":
	main()
