# ========================================================
# 离线多说话人转写 + DeepSeek 优化
# 需要修改的参数在 "0. 用户配置" 区域
# AUDIO_PATH: 输入音频文件路径（支持多种格式，如 mp4、wav、mp3 等）
# DEVICE: 运行设备，"cuda" 或 "cpu"
# NGPU: 使用 GPU 数量，仅当 DEVICE="cuda" 时有效
# NCPU: 使用 CPU 核数，仅当 DEVICE="cpu" 时有效
# OFFICIAL_DEEPSEEK_API_KEY: 官方 DeepSeek API Key，建议用环境变量设置
# =========================================================

import os
import json
import ffmpeg
from funasr import AutoModel
import time
import requests

# =========================================================
# 0. 用户配置
# =========================================================

AUDIO_PATH = "SVID_20260115_150848_1.mp4"
OUTPUT_JSON = "output.json"
SUMMARY_TXT = "summary.txt"

DEVICE = "cuda"
NGPU = 1
NCPU = 4

MERGE_CONTINUOUS_SPK = True
MERGE_GAP_MS = 300  # 关键参数：300ms

USE_LOCAL_DEEPSEEK = False  # True=本地；False=官方

# ---- 本地部署 ----
LOCAL_DEEPSEEK_URL = "http://192.168.10.90:11434/api/generate"
LOCAL_DEEPSEEK_MODEL = "deepseek-r1:32b"

# ---- 官方API ----
OFFICIAL_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OFFICIAL_DEEPSEEK_MODEL = "deepseek-chat"  # 或 "deepseek-reasoner"
OFFICIAL_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-f01e298e09304c28b72045d4de42100c")  # 建议用环境变量


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
# 6. 识别结果保存
# =========================================================

def print_units(units):
	print("\n===== ASR 输出（时间顺序 + spk） =====\n")
	for u in units:
		print(
			f"[{u['start_ms'] / 1000:07.2f} - {u['end_ms'] / 1000:07.3f}]"
			f"[SPK{u['spk']}] {u['text']}"
		)


def save_units_as_json(units, path):
	with open(path, "w", encoding="utf-8") as f:
		json.dump({"segments": units}, f, ensure_ascii=False, indent=2)
	print(f"\n✅ JSON 已保存：{path}")


def transfer_units_to_txt(units):
	lines = []
	for u in units:
		line = f"[{u['start_ms'] / 1000:07.3f} - {u['end_ms'] / 1000:07.3f}] [SPK{u['spk']}] {u['text']}"
		lines.append(line)
	return "\n".join(lines)


# =========================================================
# 本地调用（Ollama/私有部署）：/api/generate
# =========================================================
def _deepseek_local(prompt: str) -> str:
	payload = {
		"model" : LOCAL_DEEPSEEK_MODEL,
		"prompt": prompt,
		"stream": False,
	}
	headers = {"Content-Type": "application/json"}
	
	resp = requests.post(
		LOCAL_DEEPSEEK_URL,
		headers=headers,
		json=payload,
		timeout=MERGE_GAP_MS,
	)
	resp.raise_for_status()
	data = resp.json()
	return data.get("response", "")


# =========================================================
# 官方调用：/v1/chat/completions
# =========================================================
def _deepseek_official(prompt: str) -> str:
	if not OFFICIAL_DEEPSEEK_API_KEY:
		raise RuntimeError(
			"未检测到官方 API Key。\n"
			"请先设置环境变量 DEEPSEEK_API_KEY。\n"
			"Windows PowerShell:\n"
			"  $env:DEEPSEEK_API_KEY='sk-xxx'\n"
			"Linux/macOS:\n"
			"  export DEEPSEEK_API_KEY='sk-xxx'\n"
		)
	
	url = f"{OFFICIAL_DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
	headers = {
		"Content-Type" : "application/json",
		"Authorization": f"Bearer {OFFICIAL_DEEPSEEK_API_KEY}",
	}
	payload = {
		"model"      : OFFICIAL_DEEPSEEK_MODEL,
		"messages"   : [
			{"role": "system", "content": "你是一个专业的会议纪要整理助手。输出结构清晰、要点明确。"},
			{"role": "user", "content": prompt},
		],
		"temperature": 0.3,
		"max_tokens" : 2048,
		"stream"     : False,
	}
	
	resp = requests.post(url, headers=headers, json=payload, timeout=MERGE_GAP_MS)
	
	if resp.status_code != 200:
		# 尽量把错误细节打印出来
		try:
			err = resp.json()
			raise RuntimeError(
				f"HTTP {resp.status_code}\n{json.dumps(err, ensure_ascii=False, indent=2)}"
			)
		except Exception:
			raise RuntimeError(f"HTTP {resp.status_code}\n{resp.text}")
	
	data = resp.json()
	try:
		return data["choices"][0]["message"]["content"]
	except Exception:
		raise RuntimeError(
			"官方返回结构解析失败，原始返回如下：\n"
			f"{json.dumps(data, ensure_ascii=False, indent=2)}"
		)


# =========================================================
# 统一入口：deepseek_optimize
# =========================================================
def deepseek_optimize(prompt: str, path=SUMMARY_TXT) -> str:
	summary = ""
	if USE_LOCAL_DEEPSEEK:
		summary = _deepseek_local(prompt)
	else:
		summary = _deepseek_official(prompt)
	with open(path, "w", encoding="utf-8") as f:
		f.write(summary)
	print(f"\n✅ SUMMARY 已保存：{path}")
	return summary


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
	raw_text = transfer_units_to_txt(units)
	
	prompt = (
			"请将下面口语化内容整理为正式、结构化的会议纪要（包含：议题、结论、行动项）：\n\n"
			+ raw_text
	)
	
	start = time.time()
	try:
		out = deepseek_optimize(prompt)
	except Exception as e:
		print("❌ 调用失败：")
		print(e)
		return
	
	print("\n===== DeepSeek 输出 =====\n")
	print(out)
	print("\n========================")
	elapsed = time.time() - start
	print(f"⏱️ DeepSeek 耗时：{elapsed:.2f} 秒")


if __name__ == "__main__":
	main()
