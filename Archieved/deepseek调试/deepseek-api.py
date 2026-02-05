#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import requests

# =========================
# 配置区（官方 API）
# =========================
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-f01e298e09304c28b72045d4de42100c")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TIMEOUT_SEC = int(os.getenv("DEEPSEEK_TIMEOUT", "300"))


def deepseek_chat(prompt: str) -> str:
	"""
	调用 DeepSeek 官方 Chat Completions API
	返回：choices[0].message.content
	"""
	if not DEEPSEEK_API_KEY:
		raise RuntimeError(
			"未检测到 DEEPSEEK_API_KEY 环境变量。\n"
			"Windows PowerShell:  $env:DEEPSEEK_API_KEY='sk-xxx'\n"
			"Linux/macOS:         export DEEPSEEK_API_KEY='sk-xxx'"
		)
	
	url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
	headers = {
		"Content-Type" : "application/json",
		"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
	}
	
	payload = {
		"model"      : DEEPSEEK_MODEL,
		"messages"   : [
			{"role": "system", "content": "你是一个专业的会议纪要整理助手。输出结构清晰、要点明确。"},
			{"role": "user", "content": prompt},
		],
		"temperature": 0.3,
		"max_tokens" : 2048,
		"stream"     : False,
	}
	
	resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SEC)
	
	# 失败时尽量把服务端返回打印出来，方便你排错
	if resp.status_code != 200:
		try:
			err_json = resp.json()
			raise RuntimeError(
				f"HTTP {resp.status_code}\n"
				f"Response JSON:\n{json.dumps(err_json, ensure_ascii=False, indent=2)}"
			)
		except Exception:
			raise RuntimeError(
				f"HTTP {resp.status_code}\n"
				f"Response Text:\n{resp.text}"
			)
	
	data = resp.json()
	
	# 正常返回解析
	try:
		return data["choices"][0]["message"]["content"]
	except Exception:
		# 兜底：把原始 JSON 打出来
		raise RuntimeError(
			"返回结构解析失败，原始返回如下：\n"
			f"{json.dumps(data, ensure_ascii=False, indent=2)}"
		)


def main():
	f = open('../../video_processor/output.json', 'r', encoding='utf-8')
	data = json.load(f)
	f.close()
	print(data)
	full_text = ''.join([segment['text'] for segment in data['segments']])
	prompt = (
		"请将下面口语化内容整理为正式、结构化的会议纪要（包含：议题、结论、行动项）：\n\n"
		+full_text
	)
	
	print("=== DeepSeek Official API Test ===")
	print(f"Base URL : {DEEPSEEK_BASE_URL}")
	print(f"Model    : {DEEPSEEK_MODEL}")
	print(f"Timeout  : {TIMEOUT_SEC}s")
	print("----------------------------------")
	
	start = time.time()
	result = deepseek_chat(prompt)
	elapsed = time.time() - start
	
	print("\n===== DeepSeek 返回结果 =====\n")
	print(result)
	print("\n============================")
	print(f"⏱️ 耗时：{elapsed:.2f} 秒")


if __name__ == "__main__":
	main()
