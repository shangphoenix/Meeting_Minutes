# spk-RT&Offline.py
# 整合了实时 ASR 和离线 spk-cam++.py 的功能。
# 实时部分进行 VAD 切段并打印 ASR 结果，录音结束后进行离线的说话人聚类和最终输出。
# =========================================================
import os
import json
import wave
from typing import List, Dict, Any

import numpy as np
import pyaudio
import ffmpeg
from funasr import AutoModel

# =========================================================
# 输出（按你的要求）
# =========================================================
FULL_WAV_PATH = "output/session_full.wav"
OUTPUT_JSON = "output/stream_output.json"
DEBUG_RAW_JSON = "output/debug_segments_raw.json"

# =========================================================
# 录音/实时参数
# =========================================================
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024

# 实时 VAD chunk（ms）
RT_VAD_CHUNK_MS = 300
RT_MAX_END_SILENCE_MS = 500

# 实时 ASR（只用于实时打印，不做说话人区分）
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


def offline_postprocess(full_wav_path: str):
	print("\n🧾 Offline: VAD → ASR → PUNC → Cam++ (spk-cam++.py style) ...")
	audio_bytes = load_audio_bytes(full_wav_path)
	
	# 你原脚本默认 device="cuda", ngpu=1, ncpu=4，这里保持一致 :contentReference[oaicite:3]{index=3}
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
# 实时：VAD → ASR 打印（不做 speaker）
# （保持你当前“实时字幕”的需求）
# =========================================================
def main():
	# 实时模型（只 VAD+ASR）
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
	
	# 录音设备
	p = pyaudio.PyAudio()
	stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
	
	print("🎙️ 实时识别：VAD → ASR（不分speaker）。Ctrl+C 停止录音并开始离线输出（含 spk）。")
	
	# 完整音频缓存（int16 chunks，避免 np.append 低效）
	full_i16_chunks: List[np.ndarray] = []
	
	# 实时 VAD 切段状态（沿用你之前逻辑）
	chunk_size = int(RT_VAD_CHUNK_MS * RATE / 1000)
	audio_buffer = np.array([], dtype=np.float32)
	audio_vad = np.array([], dtype=np.float32)
	
	cache_vad: Dict[str, Any] = {}
	cache_asr: Dict[str, Any] = {}
	
	last_vad_beg = -1
	last_vad_end = -1
	offset_ms = 0
	
	try:
		while True:
			data = stream.read(CHUNK, exception_on_overflow=False)
			audio_i16 = np.frombuffer(data, dtype=np.int16)
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
						speech = audio_vad[beg:end]
						
						asr_out = rt_asr.generate(
							input=speech,
							cache=cache_asr,
							language="auto",
							use_itn=True,
						)
						text = ""
						if asr_out and isinstance(asr_out, list):
							text = (asr_out[0].get("text") or "").strip()
						
						seg_end_ms_global = int(offset_ms)
						seg_start_ms_global = int(offset_ms - end_ms_local)
						
						if text:
							print(f"[{seg_start_ms_global / 1000:07.3f}-{seg_end_ms_global / 1000:07.3f}] {text}")
						
						# 消费已用音频
						audio_vad = audio_vad[end:]
						last_vad_beg = -1
						last_vad_end = -1
	
	except KeyboardInterrupt:
		print("\n⏹️ 停止录音，开始保存完整音频并离线生成 spk 输出...")
	
	finally:
		stream.stop_stream()
		stream.close()
		p.terminate()
	
	# 保存完整 wav
	full_i16 = np.concatenate(full_i16_chunks, axis=0) if full_i16_chunks else np.zeros(0, dtype=np.int16)
	save_wav_pcm16(FULL_WAV_PATH, full_i16, RATE)
	print(f"✅ Full audio saved: {FULL_WAV_PATH} (samples={len(full_i16)})")
	
	# 录制完成后：按 spk-cam++.py 规范跑离线输出
	offline_postprocess(FULL_WAV_PATH)


if __name__ == "__main__":
	# main()
	offline_postprocess("../test.wav")
