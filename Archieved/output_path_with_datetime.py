import os
from datetime import datetime

def make_output_session(base_dir="output"):
    now = datetime.now()
    session_name = now.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_dir, session_name)

    os.makedirs(session_dir, exist_ok=True)

    paths = {
        "base": session_dir,
        "wav":  os.path.join(session_dir, "session_full.wav"),
        "json": os.path.join(session_dir, "stream_output.json"),
        "debug": os.path.join(session_dir, "debug_segments_raw.json"),
    }
    return paths


if __name__ == "__main__":
    p = make_output_session()

    print("输出目录:", p["base"])
    print("WAV:", p["wav"])
    print("JSON:", p["json"])
    print("DEBUG:", p["debug"])
