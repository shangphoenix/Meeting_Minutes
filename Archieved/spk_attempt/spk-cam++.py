import os
import json
import ffmpeg
from funasr import AutoModel

# =========================================================
# 0. 用户配置
# =========================================================

AUDIO_PATH = "../test.wav"
OUTPUT_JSON = "output/spk-cam++-output.json"

DEVICE = "cuda"
NGPU = 1
NCPU = 4

MERGE_CONTINUOUS_SPK = True
MERGE_GAP_MS = 300  # 关键参数：300ms


# =========================================================
# 1. 模型路径
# =========================================================

def get_model_paths():
	home = os.path.expanduser("~")
	base = os.path.join(home, ".cache", "modelscope", "hub", "models", "iic")
	return {
		"asr" : os.path.join(base, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
		"vad" : os.path.join(base, "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
		"punc": os.path.join(base, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
		"spk" : os.path.join(base, "speech_campplus_sv_zh-cn_16k-common"),
	}


# =========================================================
# 2. 音频加载
# =========================================================

def load_audio(audio_path: str) -> bytes:
	audio_bytes, _ = (
		ffmpeg
		.input(audio_path, threads=0)
		.output("-", format="wav", acodec="pcm_s16le", ac=1, ar=16000)
		.run(cmd=["ffmpeg", "-nostdin"],
		     capture_stdout=True,
		     capture_stderr=True)
	)
	return audio_bytes


# =========================================================
# 3. 模型加载
# =========================================================

def load_asr_model(device, ngpu, ncpu):
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
		disable_update=True
	)


# =========================================================
# 4. ASR
# =========================================================

def run_asr(model, audio_bytes):
	res = model.generate(
		input=audio_bytes,
		batch_size_s=300,
		is_final=True,
		sentence_timestamp=True
	)
	return res[0]


# =========================================================
# 5. 构建输出单元（带 gap）
# =========================================================

def build_output_units(sentence_info, merge_continuous_spk, merge_gap_ms):
	units = []
	
	for s in sentence_info:
		text = s["text"].strip()
		if not text:
			continue
		
		spk = s["spk"]
		start = s["start"]
		end = s["end"]
		
		if merge_continuous_spk and units:
			last = units[-1]
			same_spk = last["spk"] == spk
			gap = start - last["end_ms"]
			
			if same_spk and gap <= merge_gap_ms:
				last["text"] += text
				last["end_ms"] = end
				continue
		
		units.append({
			"spk"     : spk,
			"text"    : text,
			"start_ms": start,
			"end_ms"  : end
		})
	return units


# =========================================================
# 6. 输出
# =========================================================

def print_units(units):
	print("\n===== ASR 输出（时间顺序 + spk） =====\n")
	for u in units:
		print(
			f"[{u['start_ms'] / 1000:07.3f} - {u['end_ms'] / 1000:07.3f}]"
			f"[SPK{u['spk']}] {u['text']}"
		)


def save_units_as_json(units, path):
	with open(path, "w", encoding="utf-8") as f:
		json.dump({"segments": units}, f, ensure_ascii=False, indent=2)
	print(f"\n✅ JSON 已保存：{path}")


# =========================================================
# 7. 主流程
# =========================================================

def main():
	audio_bytes = load_audio(AUDIO_PATH)
	model = load_asr_model(DEVICE, NGPU, NCPU)
	rec = run_asr(model, audio_bytes)
	
	units = build_output_units(
		rec["sentence_info"],
		MERGE_CONTINUOUS_SPK,
		MERGE_GAP_MS
	)
	
	print_units(units)
	save_units_as_json(units, OUTPUT_JSON)


if __name__ == "__main__":
	main()
