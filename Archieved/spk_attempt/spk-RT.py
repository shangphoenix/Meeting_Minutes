import numpy as np
import pyaudio
from funasr import AutoModel

# ================== 参数 ==================
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024
CHUNK_MS = 300

# ================== 模型 ==================
model_asr = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    device="cuda:0",
    disable_update=True
)

model_vad = AutoModel(
    model="fsmn-vad",
    model_revision="v2.0.4",
    disable_pbar=True,
    max_end_silence_time=500,
    disable_update=True
)

# ================== PyAudio ==================
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

print("🎙️ PyAudio streaming start (Ctrl+C to stop)")

# ================== 状态 ==================
chunk_size = int(CHUNK_MS * RATE / 1000)
audio_buffer = np.array([], dtype=np.float32)
audio_vad = np.array([], dtype=np.float32)

cache_vad = {}
cache_asr = {}

last_vad_beg = last_vad_end = -1
offset = 0

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

            if len(res_vad[0]["value"]) > 0:
                for seg in res_vad[0]["value"]:
                    if seg[0] > -1:
                        last_vad_beg = seg[0]
                    if seg[1] > -1:
                        last_vad_end = seg[1]

                    if last_vad_beg > -1 and last_vad_end > -1:
                        last_vad_beg -= offset
                        last_vad_end -= offset
                        offset += last_vad_end

                        beg = int(last_vad_beg * RATE / 1000)
                        end = int(last_vad_end * RATE / 1000)

                        speech = audio_vad[beg:end]

                        # ===== ASR =====
                        result = model_asr.generate(
                            input=speech,
                            cache=cache_asr,
                            language="auto",
                            use_itn=True,
                        )

                        if result and result[0].get("text"):
                            print("📝", result[0]["text"])

                        audio_vad = audio_vad[end:]
                        last_vad_beg = last_vad_end = -1

except KeyboardInterrupt:
    print("\n⏹️ stopped")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
