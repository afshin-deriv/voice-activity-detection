import sounddevice as sd
import numpy as np
from typing import Optional, Callable, List
import logging
from queue import Queue, Empty
from threading import Thread, Event
from dataclasses import dataclass
from .detector import VADConfig, VoiceActivityDetector, VADResult

logger = logging.getLogger(__name__)

@dataclass
class AudioStreamConfig:
    """Configuration for audio stream processing"""
    device_name: Optional[str] = None
    input_channels: int = 1
    buffer_size: int = 2048  # Increased for better quality
    queue_size: int = 4000   # Increased for longer buffer
    dtype: np.dtype = np.float32

class AudioStreamHandler:
    """Handles real-time audio streaming and VAD processing"""
    
    def __init__(self, vad_config: VADConfig, stream_config: Optional[AudioStreamConfig] = None):
        self.vad_config = vad_config
        self.stream_config = stream_config or AudioStreamConfig()
        
        # Initialize audio components
        self._audio_queue = Queue(maxsize=self.stream_config.queue_size)
        self._stop_event = Event()
        self._stream: Optional[sd.InputStream] = None
        
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Callback for audio stream processing"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        try:
            # Extract mono audio and convert
            audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
            
            # Apply gain for better volume
            audio_chunk = audio_chunk * 1.5
            audio_chunk = np.clip(audio_chunk, -1, 1)
            
            if audio_chunk.dtype != self.stream_config.dtype:
                audio_chunk = audio_chunk.astype(self.stream_config.dtype)
            
            # Try to add to queue, drop if full
            try:
                self._audio_queue.put_nowait(audio_chunk)
            except Queue.Full:
                logger.warning("Audio queue full, dropping chunk")
                
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            
    def start(self) -> None:
        """Start audio streaming"""
        if self._stream is not None:
            logger.warning("Stream already running")
            return
            
        try:
            logger.info("Starting audio stream...")
            
            # List available devices
            devices = sd.query_devices()
            logger.info(f"Available audio devices: {devices}")
            
            # Create input stream
            self._stream = sd.InputStream(
                device=self.stream_config.device_name,
                channels=self.stream_config.input_channels,
                samplerate=self.vad_config.sample_rate,
                blocksize=self.stream_config.buffer_size,
                dtype=self.stream_config.dtype,
                callback=self._audio_callback,
                latency='low'  # Lower latency for better real-time performance
            )
            
            # Clear stop event
            self._stop_event.clear()
            
            # Start stream
            self._stream.start()
            logger.info("Audio stream started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.stop()
            raise
            
    def stop(self) -> None:
        """Stop audio streaming"""
        logger.info("Stopping audio stream...")
        
        # Set stop event
        self._stop_event.set()
        
        # Stop stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
            finally:
                self._stream = None
                
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except Empty:
                break
                
        logger.info("Audio stream stopped")
        
    @property
    def is_running(self) -> bool:
        """Check if the stream is currently running"""
        return self._stream is not None and self._stream.active
        
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()