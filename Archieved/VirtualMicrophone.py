import pyaudio
import wave
import numpy as np
import threading
import time


class VirtualMicrophone:
	def __init__(self, wav_file, chunk_size=1024, sample_rate=44100):
		"""
		创建虚拟麦克风，从WAV文件提供音频数据
		"""
		self.wav_file = wav_file
		self.chunk_size = chunk_size
		self.sample_rate = sample_rate
		self.is_running = False
		self.thread = None
		
		# 打开WAV文件
		self.wf = wave.open(wav_file, 'rb')
		
		# 获取音频参数
		self.original_rate = self.wf.getframerate()
		self.channels = self.wf.getnchannels()
		self.sample_width = self.wf.getsampwidth()
		
		# 读取所有数据
		self.audio_data = self.wf.readframes(self.wf.getnframes())
		self.wf.close()
		
		# 转换为numpy数组以便处理
		if self.sample_width == 2:
			self.audio_array = np.frombuffer(self.audio_data, dtype=np.int16)
		else:
			self.audio_array = np.frombuffer(self.audio_data, dtype=np.int8)
		
		# 如果需要重采样
		if self.original_rate != sample_rate:
			self._resample_audio()
		
		self.current_position = 0
	
	def _resample_audio(self):
		"""重采样音频到目标采样率"""
		from scipy import signal
		
		num_samples = int(len(self.audio_array) * self.sample_rate / self.original_rate)
		self.audio_array = signal.resample(self.audio_array, num_samples).astype(self.audio_array.dtype)
	
	def read_chunk(self):
		"""读取一个chunk的数据"""
		start = self.current_position
		end = start + self.chunk_size
		
		if end >= len(self.audio_array):
			# 循环播放
			remaining = self.chunk_size
			chunk = np.array([], dtype=self.audio_array.dtype)
			
			while remaining > 0:
				read_end = min(self.current_position + remaining, len(self.audio_array))
				part = self.audio_array[self.current_position:read_end]
				chunk = np.concatenate((chunk, part))
				
				remaining -= len(part)
				self.current_position = (self.current_position + len(part)) % len(self.audio_array)
		else:
			chunk = self.audio_array[start:end]
			self.current_position = end
		
		return chunk.tobytes()
	
	def callback(self, in_data, frame_count, time_info, status):
		"""PyAudio回调函数，提供音频数据"""
		if self.is_running:
			data = self.read_chunk()
			return (data, pyaudio.paContinue)
		else:
			return (None, pyaudio.paComplete)
	
	def start(self):
		"""启动虚拟麦克风"""
		self.p = pyaudio.PyAudio()
		
		# 打开输入流（实际上是输出数据给其他程序）
		self.stream = self.p.open(
			format=self.p.get_format_from_width(self.sample_width),
			channels=self.channels,
			rate=self.sample_rate,
			frames_per_buffer=self.chunk_size,
			output=False,  # 注意：这里设置为False，表示这是输入设备
			input=True,  # 对其他程序来说是输入
			stream_callback=self.callback
		)
		
		self.is_running = True
		self.stream.start_stream()
		print("虚拟麦克风已启动...")
		
		# 保持运行
		while self.is_running and self.stream.is_active():
			time.sleep(0.1)
	
	def stop(self):
		"""停止虚拟麦克风"""
		self.is_running = False
		if hasattr(self, 'stream'):
			self.stream.stop_stream()
			self.stream.close()
		if hasattr(self, 'p'):
			self.p.terminate()
		print("虚拟麦克风已停止")


# 使用示例
if __name__ == "__main__":
	mic = VirtualMicrophone("test.wav", chunk_size=1024)
	
	try:
		# 启动虚拟麦克风
		mic.start()
	except KeyboardInterrupt:
		mic.stop()