import json

f = open("output/20260106_191752/stream_output.json", "r", encoding="utf-8")
data = json.load(f)
f.close()
for segment in data["segments"]:
	spk = segment["spk"] + 1
	start = segment["start_ms"] / 1000.0
	end = segment["end_ms"] / 1000.0
	text = segment["text"]
	# 输出格式为：
	# [speaker:spk][start_time - end_time]: text
	print(f"[speaker:{spk:2}]\t[{start:^8.2f}-{end:^8.2f}]: {text}")
