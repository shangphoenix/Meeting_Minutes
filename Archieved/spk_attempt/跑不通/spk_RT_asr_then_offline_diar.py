import os
import json
import wave
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pyaudio
import torch
from funasr import AutoModel


# =========================
# 参数
# =========================
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024

# 实时 VAD chunk (ms)
RT_VAD_CHUNK_MS = 300
RT_MAX_END_SILENCE_MS = 500

# 离线 VAD chunk (ms)（可同实时）
OFF_VAD_CHUNK_MS = 300
OFF_MAX_END_SILENCE_MS = 500

# 模型
ASR_MODEL = "iic/SenseVoiceSmall"
VAD_MODEL = "fsmn-vad"
VAD_REV = "v2.0.4"
SPK_MODEL = "iic/speech_campplus_sv_zh-cn_16k-common"

# 设备（按需改）
ASR_DEVICE = "cuda:0"
VAD_DEVICE = "cuda:0"
SPK_DEVICE = "cuda:0"

# 输出
FULL_WAV_PATH = "../output/session_full.wav"
OUTPUT_JSON = "output/stream_output.json"
DEBUG_RAW_JSON = "output/debug_segments_raw.json"  # 离线 VAD+ASR 的细段调试

# 声纹/聚类参数
SPK_MIN_EMB_MS = 1500
SPK_COS_THRESH = 0.70

# 声纹稳定：先把相邻段拼成更长块再算 emb
SPK_PREMERGE_GAP_MS = 800
SPK_PREMERGE_MIN_MS = 2500

# 最终合并输出（同 spk 且 gap 小就拼）
FINAL_MERGE_GAP_MS = 400


# =========================
# 工具：WAV 读写
# =========================
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_wav_pcm16(path: str, audio_i16: np.ndarray, sr: int = 16000):
    ensure_dir(path)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())

def load_wav_pcm16(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if ch != 1 or sw != 2:
        raise ValueError(f"Only mono PCM16 supported. got ch={ch}, sw={sw}")
    audio_i16 = np.frombuffer(raw, dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32767.0
    return audio_f32, sr


# =========================
# Speaker clustering
# =========================
def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x)) + eps
    return x / n

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_l2norm(a), _l2norm(b)))

@dataclass
class SpeakerAssigner:
    cos_thresh: float = SPK_COS_THRESH
    centroids: List[np.ndarray] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)

    def assign(self, emb: np.ndarray) -> int:
        emb = _l2norm(np.asarray(emb, dtype=np.float32).reshape(-1))
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
# Cam++ embedding 提取（兼容 CUDA tensor）
# =========================
def _to_numpy(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    return np.asarray(x, dtype=np.float32).reshape(-1)

def extract_spk_embedding(model: AutoModel, speech_f32: np.ndarray) -> Optional[np.ndarray]:
    try:
        out = model.generate(input=speech_f32)
        if not out:
            return None
        obj = out[0] if isinstance(out, list) else out

        keys = ["spk_embedding", "speaker_embedding", "embedding", "emb", "spk_emb", "xvector", "vector"]

        if torch.is_tensor(obj) or isinstance(obj, (list, tuple, np.ndarray)):
            return _to_numpy(obj)

        if isinstance(obj, dict):
            for k in keys:
                if k in obj and obj[k] is not None:
                    return _to_numpy(obj[k])

            for nest_k in ["value", "outputs", "result", "data"]:
                v = obj.get(nest_k)
                if isinstance(v, dict):
                    for k in keys:
                        if k in v and v[k] is not None:
                            return _to_numpy(v[k])
                elif v is not None and (torch.is_tensor(v) or isinstance(v, (list, tuple, np.ndarray))):
                    arr = _to_numpy(v)
                    if arr is not None and 8 <= arr.size <= 4096:
                        return arr

            for v in obj.values():
                if v is None:
                    continue
                if torch.is_tensor(v) or isinstance(v, (list, tuple, np.ndarray)):
                    arr = _to_numpy(v)
                    if arr is not None and 8 <= arr.size <= 4096:
                        return arr

        return None
    except Exception as e:
        print(f"[WARN] extract_spk_embedding failed: {e}")
        return None


# =========================
# 离线：VAD（流式扫完整音频，取 (start_ms,end_ms)）
# =========================
def vad_segments_streaming(model_vad: AutoModel, audio: np.ndarray, sr: int, chunk_ms: int) -> List[Tuple[int, int]]:
    chunk_size = int(chunk_ms * sr / 1000)
    cache_vad: Dict[str, Any] = {}

    last_beg = -1
    last_end = -1
    offset_ms = 0
    audio_vad = np.array([], dtype=np.float32)

    segs: List[Tuple[int, int]] = []
    i = 0
    while i < len(audio):
        chunk = audio[i:i+chunk_size]
        i += chunk_size
        if len(chunk) == 0:
            break

        audio_vad = np.append(audio_vad, chunk)

        res = model_vad.generate(input=chunk, cache=cache_vad, is_final=False, chunk_size=chunk_ms)
        values = []
        if res and isinstance(res, list) and res[0].get("value") is not None:
            values = res[0]["value"]
        if not values:
            continue

        for seg in values:
            if seg[0] > -1:
                last_beg = seg[0]
            if seg[1] > -1:
                last_end = seg[1]

            if last_beg > -1 and last_end > -1:
                beg_ms_local = last_beg - offset_ms
                end_ms_local = last_end - offset_ms
                offset_ms += end_ms_local

                beg = int(beg_ms_local * sr / 1000)
                end = int(end_ms_local * sr / 1000)

                seg_end_ms_global = int(offset_ms)
                seg_start_ms_global = int(offset_ms - end_ms_local)
                segs.append((seg_start_ms_global, seg_end_ms_global))

                audio_vad = audio_vad[end:]
                last_beg, last_end = -1, -1

    # flush（可选）
    try:
        model_vad.generate(input=np.zeros(0, dtype=np.float32), cache=cache_vad, is_final=True)
    except Exception:
        pass

    return segs


# =========================
# 离线：ASR（对 VAD 段逐段识别）
# =========================
def asr_on_segments(model_asr: AutoModel, audio: np.ndarray, segs_ms: List[Tuple[int, int]], sr: int) -> List[Dict[str, Any]]:
    cache_asr: Dict[str, Any] = {}
    out = []
    for idx, (s_ms, e_ms) in enumerate(segs_ms):
        s = int(s_ms * sr / 1000)
        e = int(e_ms * sr / 1000)
        speech = audio[s:e]
        if len(speech) == 0:
            continue
        r = model_asr.generate(input=speech, cache=cache_asr, language="auto", use_itn=True)
        text = ""
        if r and isinstance(r, list):
            text = (r[0].get("text") or "").strip()
        out.append({
            "id": idx,
            "start_ms": int(s_ms),
            "end_ms": int(e_ms),
            "text": text,
            "audio": speech.astype(np.float32),
        })
    return out


# =========================
# 离线：先拼长块算声纹，再回填到细段
# =========================
def premerge_for_spk(raw: List[Dict[str, Any]], gap_ms: int, min_ms: int, sr: int) -> List[Dict[str, Any]]:
    merged = []
    cur = None
    for seg in raw:
        if cur is None:
            cur = {"start_ms": seg["start_ms"], "end_ms": seg["end_ms"], "audio": seg["audio"], "children_ids": [seg["id"]]}
            continue
        gap = seg["start_ms"] - cur["end_ms"]
        if gap <= gap_ms:
            cur["end_ms"] = seg["end_ms"]
            cur["audio"] = np.concatenate([cur["audio"], seg["audio"]], axis=0)
            cur["children_ids"].append(seg["id"])
        else:
            merged.append(cur)
            cur = {"start_ms": seg["start_ms"], "end_ms": seg["end_ms"], "audio": seg["audio"], "children_ids": [seg["id"]]}
    if cur is not None:
        merged.append(cur)

    for m in merged:
        m["dur_ms"] = int(len(m["audio"]) * 1000 / sr)
        m["ok"] = m["dur_ms"] >= min_ms
    return merged

def assign_spk_offline(raw: List[Dict[str, Any]], model_spk: AutoModel, sr: int) -> List[int]:
    spk_ids = [-1] * len(raw)
    merged = premerge_for_spk(raw, gap_ms=SPK_PREMERGE_GAP_MS, min_ms=SPK_PREMERGE_MIN_MS, sr=sr)
    assigner = SpeakerAssigner(cos_thresh=SPK_COS_THRESH)

    for m in merged:
        spk = -1
        if m["dur_ms"] >= SPK_MIN_EMB_MS and m["ok"]:
            emb = extract_spk_embedding(model_spk, m["audio"])
            if emb is not None:
                spk = assigner.assign(emb)
        for cid in m["children_ids"]:
            if 0 <= cid < len(spk_ids):
                spk_ids[cid] = spk

    # 兜底：把仍为 -1 的段用“最近已知 spk”填充（常用且有效）
    last = -1
    for i in range(len(spk_ids)):
        if spk_ids[i] != -1:
            last = spk_ids[i]
        else:
            spk_ids[i] = last
    return spk_ids


# =========================
# 离线：最终合并输出
# =========================
def final_merge(raw: List[Dict[str, Any]], spk_ids: List[int], gap_ms: int) -> List[Dict[str, Any]]:
    out = []
    for seg in raw:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        sid = spk_ids[seg["id"]] if seg["id"] < len(spk_ids) else -1
        s_ms, e_ms = seg["start_ms"], seg["end_ms"]

        if out:
            last = out[-1]
            if last["spk"] == sid and (s_ms - last["end_ms"]) <= gap_ms:
                last["text"] += text
                last["end_ms"] = e_ms
                continue

        out.append({"spk": int(sid), "text": text, "start_ms": int(s_ms), "end_ms": int(e_ms)})
    return out


# =========================
# 主流程：实时 VAD->ASR（打印） + 录完整音频；结束后离线 VAD->ASR->Cam++...
# =========================
def main():
    # --- 实时模型（只 VAD+ASR） ---
    rt_vad = AutoModel(
        model=VAD_MODEL,
        model_revision=VAD_REV,
        disable_pbar=True,
        max_end_silence_time=RT_MAX_END_SILENCE_MS,
        device=VAD_DEVICE,
        disable_update=True
    )
    rt_asr = AutoModel(
        model=ASR_MODEL,
        trust_remote_code=True,
        device=ASR_DEVICE,
        disable_update=True
    )

    # --- 录音设备 ---
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("🎙️ 实时字幕开始：VAD → ASR（不分speaker）。按 Ctrl+C 结束并开始离线分speaker。")

    # 录完整音频：用 int16 列表拼接，避免 np.append 很慢
    full_i16_chunks: List[np.ndarray] = []

    # 实时 VAD/ASR 状态（基本沿用你之前 realtime 逻辑）
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

            # 1) 完整音频缓存（关键）
            full_i16_chunks.append(audio_i16.copy())

            # 2) 实时 VAD+ASR 用 float32
            audio_f32 = audio_i16.astype(np.float32) / 32767.0
            audio_buffer = np.append(audio_buffer, audio_f32)

            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]
                audio_vad = np.append(audio_vad, chunk)

                # VAD streaming
                res_vad = rt_vad.generate(input=chunk, cache=cache_vad, is_final=False, chunk_size=RT_VAD_CHUNK_MS)

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

                        # ASR
                        asr_out = rt_asr.generate(input=speech, cache=cache_asr, language="auto", use_itn=True)
                        text = ""
                        if asr_out and isinstance(asr_out, list):
                            text = (asr_out[0].get("text") or "").strip()

                        seg_end_ms_global = int(offset_ms)
                        seg_start_ms_global = int(offset_ms - end_ms_local)

                        if text:
                            print(f"[{seg_start_ms_global/1000:07.3f}-{seg_end_ms_global/1000:07.3f}] {text}")

                        # 消费已用音频（与实时逻辑一致）
                        audio_vad = audio_vad[end:]
                        last_vad_beg = -1
                        last_vad_end = -1

    except KeyboardInterrupt:
        print("\n⏹️ 录音结束，开始保存完整音频并离线分speaker...")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    # --- 保存完整音频 wav ---
    full_i16 = np.concatenate(full_i16_chunks, axis=0) if full_i16_chunks else np.zeros(0, dtype=np.int16)
    save_wav_pcm16(FULL_WAV_PATH, full_i16, RATE)
    print(f"✅ 完整录音已保存：{FULL_WAV_PATH} (samples={len(full_i16)})")

    # --- 离线阶段：重新跑 VAD → ASR → Cam++ → 聚类 → 合并 ---
    # audio_f32, sr = load_wav_pcm16(FULL_WAV_PATH)
    audio_f32, sr = load_wav_pcm16("../../test.wav")
    if sr != RATE:
        raise ValueError(f"Expected {RATE}Hz, got {sr}Hz")

    off_vad = AutoModel(
        model=VAD_MODEL,
        model_revision=VAD_REV,
        disable_pbar=True,
        max_end_silence_time=OFF_MAX_END_SILENCE_MS,
        device=VAD_DEVICE,
        disable_update=True
    )
    off_asr = AutoModel(
        model=ASR_MODEL,
        trust_remote_code=True,
        device=ASR_DEVICE,
        disable_update=True
    )
    off_spk = AutoModel(
        model=SPK_MODEL,
        device=SPK_DEVICE,
        disable_update=True
    )

    segs_ms = vad_segments_streaming(off_vad, audio_f32, sr, chunk_ms=OFF_VAD_CHUNK_MS)
    print(f"✅ 离线 VAD 段数：{len(segs_ms)}")

    raw = asr_on_segments(off_asr, audio_f32, segs_ms, sr)
    print(f"✅ 离线 ASR 段数：{len(raw)}")

    with open(DEBUG_RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(
            [{"id": r["id"], "start_ms": r["start_ms"], "end_ms": r["end_ms"], "text": r["text"]} for r in raw],
            f, ensure_ascii=False, indent=2
        )

    spk_ids = assign_spk_offline(raw, off_spk, sr)
    n_spk = (max(spk_ids) + 1) if spk_ids else 0
    print(f"✅ 离线 speaker 聚类完成：{n_spk} speakers（-1 已用最近值回填）")

    final_segments = final_merge(raw, spk_ids, gap_ms=FINAL_MERGE_GAP_MS)

    out_obj = {
        "audio_full": {"path": FULL_WAV_PATH, "sample_rate": RATE, "channels": 1},
        "segments": final_segments
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"✅ 最终输出：{OUTPUT_JSON}")
    print(f"   - Debug raw: {DEBUG_RAW_JSON}")


if __name__ == "__main__":
    main()
