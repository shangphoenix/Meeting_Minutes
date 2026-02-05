#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSeek API integrated test js
- USE_LOCAL_DEEPSEEK = True  -> Local Ollama-like /api/generate
- USE_LOCAL_DEEPSEEK = False -> Official DeepSeek /v1/chat/completions
"""

import os
import time
import json
import requests

# =========================================================
# 配置区：一键切换
# =========================================================
USE_LOCAL_DEEPSEEK = False  # <<< True=本地；False=官方

# ---- 本地 ----
LOCAL_DEEPSEEK_URL = "http://192.168.10.90:11434/api/generate"
LOCAL_DEEPSEEK_MODEL = "deepseek-r1:32b"

# ---- 官方 DeepSeek ----
OFFICIAL_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OFFICIAL_DEEPSEEK_MODEL = "deepseek-chat"  # 或 "deepseek-reasoner"
OFFICIAL_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-f01e298e09304c28b72045d4de42100c")  # 建议用环境变量

# ---- 通用 ----
TIMEOUT_SEC = 300


# =========================================================
# 本地调用（Ollama/私有部署）：/api/generate
# =========================================================
def deepseek_local(prompt: str) -> str:
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
		timeout=TIMEOUT_SEC,
	)
	resp.raise_for_status()
	data = resp.json()
	return data.get("response", "")


# =========================================================
# 官方调用：/v1/chat/completions
# =========================================================
def deepseek_official(prompt: str) -> str:
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
	
	resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)
	
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
def deepseek_optimize(prompt: str) -> str:
	if USE_LOCAL_DEEPSEEK:
		return deepseek_local(prompt)
	return deepseek_official(prompt)


# =========================================================
# 测试主程序
# =========================================================
def main():
	prompt = (
		"请将下面口语化内容整理为正式、结构化的会议纪要（包含：议题、结论、行动项）：\n\n"
		"我们这次主要讨论了模型训练的稳定性问题，loss 有点波动，"
		"大家觉得可能需要调一下 learning rate，"
		"然后数据这块也要再做一次清洗，看看有没有脏标注。"
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
	print(f"⏱️ 耗时：{elapsed:.2f} 秒")


if __name__ == "__main__":
	main()
