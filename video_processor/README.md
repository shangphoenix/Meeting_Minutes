# 离线视频/音频处理模块

## 📋 模块简介
基于 FunASR 的离线多说话人语音识别与 DeepSeek AI 优化模块，支持视频/音频文件的离线处理，自动生成结构化会议纪要。

## ✨ 核心功能
- 🎬 **视频/音频文件处理**：支持 MP4、WAV、MP3 等多种格式
- 👥 **多说话人分离**：自动识别并区分不同发言人
- ⏱️ **时间戳标注**：精确到毫秒的起止时间标注
- 🤖 **AI 智能优化**：通过 DeepSeek 大模型优化会议纪要结构
- 📊 **结构化输出**：生成 JSON 格式的会议记录
- 🔧 **灵活配置**：支持本地部署和官方 API 两种 DeepSeek 调用方式

## 🛠️ 技术栈
| 模块 | 技术/模型 | 说明 |
|------|-----------|------|
| **语音识别** | FunASR (seaco-paraformer-large) | 高精度中文语音识别 |
| **语音活动检测** | fsmn-vad | 语音端点检测 |
| **标点恢复** | punc_ct-transformer | 中文标点恢复 |
| **说话人识别** | campplus | 说话人分离与识别 |
| **AI 优化** | DeepSeek API | 会议纪要结构化优化 |
| **音频处理** | FFmpeg | 音频格式转换与处理 |

## 📁 文件结构
```
video_processor/
├── spk-deepseek.py          # 主处理脚本
├── SVID_20260115_150848_1.mp4  # 示例视频文件
└── output.json              # 处理结果输出
```

## 🚀 快速使用

### 1. 环境准备
```bash
# 安装依赖（已在项目根目录 requirements.txt 中）
pip install -r ../requirements.txt

# 确保 FFmpeg 已安装并添加到 PATH
ffmpeg -version
```

### 2. 配置参数
编辑 `spk-deepseek.py` 中的用户配置区域：

```python
# =========================================================
# 0. 用户配置
# =========================================================

AUDIO_PATH = "SVID_20260115_150848_1.mp4"  # 输入文件路径
OUTPUT_JSON = "output.json"                # 输出 JSON 文件路径

DEVICE = "cuda"    # 运行设备："cuda" 或 "cpu"
NGPU = 1           # GPU 数量（仅当 DEVICE="cuda" 时有效）
NCPU = 4           # CPU 核心数（仅当 DEVICE="cpu" 时有效）

MERGE_CONTINUOUS_SPK = True   # 合并同一说话人连续片段
MERGE_GAP_MS = 300            # 合并间隔阈值（毫秒）

USE_LOCAL_DEEPSEEK = False    # True=本地部署，False=官方API

# ---- 本地部署配置 ----
LOCAL_DEEPSEEK_URL = "http://192.168.10.90:11434/api/generate"
LOCAL_DEEPSEEK_MODEL = "deepseek-r1:32b"

# ---- 官方API配置 ----
OFFICIAL_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OFFICIAL_DEEPSEEK_MODEL = "deepseek-chat"
OFFICIAL_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key")
```

### 3. 运行处理
```bash
cd video_processor
python spk-deepseek.py
```

### 4. 查看结果
处理完成后，系统会：
1. 在控制台显示带时间戳和说话人标签的转写结果
2. 生成 `output.json` 文件
3. 调用 DeepSeek 优化会议纪要结构并显示

## 📊 输出格式

### JSON 输出结构
```json
{
  "segments": [
    {
      "spk": 0,
      "text": "大家好，我们开始今天的会议。",
      "start_ms": 530,
      "end_ms": 3445
    },
    {
      "spk": 1,
      "text": "我同意这个提议。",
      "start_ms": 4500,
      "end_ms": 6200
    }
  ]
}
```

### 控制台输出示例
```
[000.53 - 003.445][SPK0] 大家好，我们开始今天的会议。
[004.50 - 006.200][SPK1] 我同意这个提议。
```

## ⚙️ 高级配置

### 模型路径配置
系统会自动从 ModelScope 下载模型，默认缓存路径为：
```
~/.cache/modelscope/hub/models/iic/
```

如需自定义模型路径，修改 `get_model_paths()` 函数。

### DeepSeek 配置选项

#### 选项1：使用官方 API（推荐）
1. 获取 DeepSeek API Key：https://platform.deepseek.com/
2. 设置环境变量：
   ```bash
   # Windows PowerShell
   $env:DEEPSEEK_API_KEY='sk-xxx'
   
   # Linux/macOS
   export DEEPSEEK_API_KEY='sk-xxx'
   ```
3. 设置 `USE_LOCAL_DEEPSEEK = False`

#### 选项2：使用本地部署（Ollama）
1. 安装并启动 Ollama：https://ollama.com/
2. 下载 DeepSeek 模型：
   ```bash
   ollama pull deepseek-r1:32b
   ```
3. 设置 `USE_LOCAL_DEEPSEEK = True`
4. 配置本地 API 地址和模型名称

## 🔧 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 模型下载失败 | 网络问题 | 检查网络连接，确保可访问 ModelScope |
| FFmpeg 错误 | FFmpeg 未安装 | 安装 FFmpeg 并添加到 PATH |
| GPU 内存不足 | 显存不够 | 减小批处理大小，或在 CPU 模式下运行 |
| DeepSeek API 失败 | API Key 错误或网络问题 | 检查 API Key，确认网络连接正常 |
| 说话人识别不准确 | 音频质量差或多人重叠 | 确保音频清晰，避免多人同时发言 |

## 📈 性能优化建议

### 硬件建议
- **GPU**：使用 NVIDIA GPU 可大幅提升处理速度
- **CPU**：多核 CPU 有助于并行处理
- **内存**：建议至少 8GB RAM

### 处理建议
1. **音频质量**：确保输入音频清晰，背景噪音小
2. **说话方式**：避免多人同时发言，系统支持但不推荐重叠语音
3. **文件格式**：推荐使用 WAV 格式以获得最佳质量
4. **批量处理**：可修改脚本支持批量文件处理

## 🔄 处理流程
1. **音频提取**：使用 FFmpeg 从视频中提取音频并转换为 16kHz WAV 格式
2. **语音识别**：使用 FunASR 进行语音转写
3. **说话人分离**：使用 campplus 模型区分不同说话人
4. **片段合并**：合并同一说话人的连续片段（可配置间隔）
5. **AI 优化**：调用 DeepSeek 优化会议纪要结构
6. **结果输出**：生成 JSON 文件和结构化会议纪要

## 📋 注意事项
1. **首次运行**：需要下载约 5GB 模型文件，请确保磁盘空间充足
2. **处理时间**：取决于音频长度和硬件性能，通常为实时音频长度的 0.5-2 倍
3. **隐私安全**：处理敏感音频时，注意数据保密
4. **API 限制**：使用官方 API 时注意调用频率和费用限制

## 🎯 适用场景
- 会议录音/视频转录
- 访谈记录整理
- 课堂讲座记录
- 播客内容转写
- 视频字幕生成

## 📞 技术支持
如遇问题，请：
1. 查看控制台错误日志
2. 检查模型文件是否完整下载
3. 确认 API Key 或本地部署配置正确
4. 提交 Issue 或联系项目维护者

---

**版本**：1.0.0 | **最后更新**：2025年1月 | **依赖**：FunASR, DeepSeek API, FFmpeg