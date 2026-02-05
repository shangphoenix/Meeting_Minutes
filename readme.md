# 智能会议纪要系统

## 📋 项目简介
一个完整的智能会议纪要解决方案，集成了实时语音识别、离线多说话人分离和AI智能优化功能。系统支持Web实时录音转写和视频/音频文件离线处理，自动生成结构化会议纪要。

## ✨ 核心特性

### 🎤 实时语音转写模块
- **实时识别**：基于SenseVoiceSmall模型，支持中英文混合识别
- **说话人分离**：录音结束后自动区分不同发言人并标注
- **时间戳标注**：每句话均附带起止时间，便于回溯与定位
- **Web交互界面**：简洁直观的前端操作，无需安装客户端
- **本地自动保存**：完整录音文件及处理结果按会话自动归档

### 🎬 离线视频/音频处理模块
- **多格式支持**：支持MP4、WAV、MP3等多种音视频格式
- **多说话人识别**：自动识别并区分不同发言人
- **AI智能优化**：通过DeepSeek大模型优化会议纪要结构
- **批量处理能力**：支持离线批量处理音视频文件

## 📁 项目文件结构

```
Meeting_Minutes/
├── README.md                          # 项目说明文档
├── app.py                             # 主后端服务（FastAPI + WebSocket）
├── index.html                         # 主前端页面
├── requirements.txt                   # Python依赖包列表
├── .env-sample                        # 环境变量示例
├── .gitignore                         # Git忽略文件配置
│
├── css/                               # 样式文件
│   └── style.css                      # 主样式文件
│
├── js/                                # JavaScript文件
│   ├── recorder.controller.js         # 录音控制器
│   ├── copy-button.js                 # 复制功能
│   ├── download-button.js             # 下载功能
│   └── jquery.js                      # jQuery库
│
├── output/                            # 输出目录（自动生成）
│   └── YYYYMMDD_HHMMSS/               # 按时间戳命名的会话目录
│       ├── session_full.wav           # 完整录音文件
│       ├── stream_output.json         # 结构化会议纪要
│       └── debug_segments_raw.json    # 原始识别片段（调试用）
│
└── video_processor/                   # 离线视频/音频处理模块
    ├── README.md                      # 模块说明文档
    ├── spk-deepseek.py                # 离线处理主脚本
    ├── output.json                    # 示例输出文件
    └── SVID_20260115_150848_1.mp4     # 示例视频文件

```

## 🛠️ 技术栈

### 后端技术
| 模块 | 技术/框架 | 说明 |
|------|-----------|------|
| **Web框架** | FastAPI + WebSocket | 高性能异步Web框架 |
| **语音识别** | FunASR | 阿里巴巴开源语音识别工具包 |
| **实时模型** | SenseVoiceSmall | 实时语音识别模型 |
| **离线模型** | seaco-paraformer-large + campplus | 高精度识别+说话人分离 |
| **音频处理** | FFmpeg + WebRTC | 音频格式转换与流处理 |

### 前端技术
| 模块 | 技术 | 说明 |
|------|------|------|
| **核心框架** | HTML5 + JavaScript | 原生Web技术 |
| **音频API** | Web Audio API + MediaRecorder | 浏览器录音功能 |
| **网络通信** | WebSocket | 实时数据传输 |
| **UI库** | jQuery | DOM操作与事件处理 |

### AI优化
| 模块 | 技术 | 说明 |
|------|------|------|
| **AI优化** | DeepSeek API | 会议纪要结构化优化 |
| **部署选项** | 官方API / 本地Ollama | 双模式支持 |

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone http://192.168.10.114:7190/ai/meeting_minutes
cd Meeting_Minutes

# 创建并激活 Conda 环境
conda create -n meeting_minutes python=3.9 -y
conda activate meeting_minutes

# 升级基础工具
python -m pip install -U pip setuptools wheel

# 安装 PyTorch（CUDA 12.8，GPU 版本）
pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# 若使用国内镜像，可改为：
# pip3 install torch torchvision torchaudio \
#   --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128/

# CPU 版本（无 GPU 时使用）：
# pip3 install torch torchvision torchaudio

# 安装项目最小依赖
pip install -r requirements.txt

# 验证 PyTorch 是否正确安装
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

```

### 2. 配置环境变量
复制环境变量示例文件并配置：
```bash
cp .env-sample .env
```
编辑 `.env` 文件，配置DeepSeek API Key等参数。

### 3. 启动实时服务
```bash
# 启动后端服务（默认端口 27000）
python app.py --host 0.0.0.0 --port 27000

# 可选参数
# --host  绑定地址 (默认: 0.0.0.0)
# --port  服务端口 (默认: 27000)
```

### 4. 使用Web界面
1. **打开前端**：浏览器访问 `http://localhost:27000` 或直接打开 `index.html`
2. **开始录音**：点击"开始录制"按钮，授予麦克风权限
3. **实时转写**：系统实时显示识别文字，带时间戳
4. **结束处理**：点击"停止录制"，系统自动进行说话人分离和AI优化
5. **查看结果**：页面显示带说话人标签的完整会议纪要和AI总结

### 5. 使用离线处理模块
```bash
# 进入离线处理目录
cd video_processor

# 配置处理参数（编辑 spk-deepseek.py）
# 修改 AUDIO_PATH、OUTPUT_JSON 等参数

# 运行离线处理
python spk-deepseek.py
```

## 📊 输出格式

### 实时模块输出
```
output/YYYYMMDD_HHMMSS/
├── session_full.wav          # 完整录音（WAV格式）
├── stream_output.json        # 结构化会议纪要（JSON）
└── debug_segments_raw.json   # 原始识别片段（调试用）
```

### JSON输出示例
```json
{
  "audio_full": {
    "path": "output/20250101_120000/session_full.wav",
    "sample_rate": 16000,
    "channels": 1
  },
  "segments": [
    {
      "spk": "0",
      "text": "大家好，我们开始今天的会议。",
      "start_ms": 0,
      "end_ms": 3200
    },
    {
      "spk": "1",
      "text": "我同意这个提议。",
      "start_ms": 4500,
      "end_ms": 6200
    }
  ]
}
```

### 离线模块输出
- **控制台输出**：带时间戳和说话人标签的转写结果
- **JSON文件**：结构化会议记录
- **AI优化结果**：DeepSeek生成的会议纪要总结

## 🔌 API接口

### WebSocket接口
| 端点 | 方法 | 说明 | 参数 |
|------|------|------|------|
| `ws://localhost:27000/ws/transcribe` | WebSocket | 实时语音转写 | `lang=auto`（语言选择） |

### HTTP接口
| 端点 | 方法 | 说明 |
|------|------|------|
| `http://localhost:27000/health` | GET | 服务健康检查 |
| `http://localhost:27000/` | GET | 前端页面服务 |

## ⚙️ 关键配置

### 实时识别参数（app.py）
```python
RATE = 16000                    # 音频采样率
RT_VAD_CHUNK_MS = 300           # VAD分块大小（毫秒）
RT_MAX_END_SILENCE_MS = 500     # 最大结束静音时间
MERGE_CONTINUOUS_SPK = True     # 合并同一说话人连续片段
MERGE_GAP_MS = 300              # 合并间隔阈值（毫秒）
```

### DeepSeek配置
支持两种模式：
1. **官方API模式**：需要配置 `DEEPSEEK_API_KEY`
2. **本地Ollama模式**：需要本地部署Ollama和DeepSeek模型

配置示例（.env文件）：
```env
# DeepSeek官方API
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 或使用本地Ollama
USE_LOCAL_DEEPSEEK=true
LOCAL_DEEPSEEK_URL=http://localhost:11434/api/generate
LOCAL_DEEPSEEK_MODEL=deepseek-r1:32b
```

## 🎯 功能对比

| 功能 | 实时模块 | 离线模块 |
|------|----------|----------|
| **输入方式** | 浏览器实时录音 | 音视频文件 |
| **处理模式** | 实时流式处理 | 离线批量处理 |
| **说话人分离** | 录音结束后处理 | 处理过程中识别 |
| **AI优化** | 支持 | 支持 |
| **输出格式** | JSON + WAV | JSON + 文本 |
| **使用场景** | 实时会议记录 | 音视频文件转录 |

## 🔧 故障排除

### 常见问题
| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 模型下载失败 | 网络问题 | 检查网络连接，确保可访问ModelScope |
| GPU内存不足 | 显存不够 | 减小批处理大小，或在CPU模式下运行 |
| WebSocket连接失败 | 端口占用/防火墙 | 检查端口27000是否可用，关闭防火墙限制 |
| 麦克风无法使用 | 浏览器权限/硬件问题 | 检查浏览器麦克风权限，确认麦克风正常工作 |
| DeepSeek API失败 | API Key错误或网络问题 | 检查API Key，确认网络连接正常 |

### 首次运行注意事项
1. **模型下载**：首次运行需要下载约5GB模型文件，请确保磁盘空间充足
2. **FFmpeg安装**：确保系统已安装FFmpeg并添加到PATH
3. **浏览器支持**：需使用Chrome/Edge等支持WebSocket和Web Audio API的现代浏览器
4. **隐私安全**：录音文件保存在本地，注意数据保密

## 📈 性能优化建议

### 硬件建议
- **GPU**：使用NVIDIA GPU可大幅提升处理速度（推荐RTX 3060+）
- **CPU**：多核CPU有助于并行处理（推荐8核+）
- **内存**：建议至少16GB RAM
- **存储**：SSD可加快模型加载速度

### 软件优化
1. **使用虚拟环境**：避免包冲突
2. **模型缓存**：首次下载后模型会缓存，后续启动更快
3. **批量处理**：离线模块支持批量处理多个文件
4. **网络优化**：确保稳定的网络连接

### 使用建议
1. **音频质量**：在安静环境下录音，识别准确率更高
2. **说话方式**：避免多人同时发言，系统支持但不推荐重叠语音
3. **文件格式**：推荐使用WAV格式以获得最佳质量
4. **实时延迟**：系统有300-500ms的处理延迟，属正常范围

## 🎯 适用场景

### 企业应用
- **会议记录**：实时记录会议内容，自动生成纪要
- **访谈转录**：采访录音转文字，区分采访者和受访者
- **培训记录**：培训课程录音转文字，便于复习
- **客户服务**：客服通话记录与分析

### 教育应用
- **课堂记录**：讲座录音转文字，生成课堂笔记
- **学术研讨**：学术讨论记录，区分不同发言者
- **在线课程**：录播课程字幕生成

### 媒体制作
- **视频字幕**：为视频自动生成字幕文件
- **播客转录**：播客内容转文字，便于搜索和引用
- **影视制作**：剧本朗读、配音录制等

## 🔄 处理流程

### 实时模块流程
```
浏览器录音 → WebSocket传输 → 实时VAD/ASR → 分段识别结果
     ↓
录音结束信号 → 保存完整WAV → 离线说话人分离
     ↓
生成JSON结果 → DeepSeek优化 → 返回AI总结
```

### 离线模块流程
```
音视频文件 → FFmpeg提取音频 → FunASR识别 → 说话人分离
     ↓
片段合并优化 → 生成JSON结果 → DeepSeek优化 → 输出结果
```

## 📋 待开发功能
- [ ] 实时说话人分离（在线阶段）
- [ ] 多语言实时翻译
- [ ] 前端实时字幕显示优化
- [ ] 导出格式扩展（Word、PDF、SRT字幕等）
- [ ] 语音指令控制
- [ ] 多房间/多会议支持
- [ ] 移动端适配
- [ ] 云存储集成

## 📞 技术支持

### 获取帮助
1. **查看日志**：检查控制台错误日志和output目录下的调试文件
2. **环境检查**：确认Python版本、依赖包、FFmpeg等环境正常
3. **模型验证**：确认模型文件完整下载

### 问题反馈
如遇问题，请提供以下信息：
1. 操作系统和Python版本
2. 错误日志或截图
3. 复现步骤
4. 相关配置文件（去除敏感信息）

### 贡献指南
欢迎提交Issue和Pull Request，共同完善项目。

---

**版本**：2.0.0 | **最后更新**：2025年1月 | **许可证**：MIT

**核心特性**：实时识别 + 离线处理 + AI优化 | **适用平台**：Windows/Linux/macOS

**项目状态**：生产可用 | **维护状态**：积极维护