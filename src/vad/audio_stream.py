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
    buffer_size: int = 1024
    queue_size: int = 1000
    dtype: np.dtype = np.float32

class AudioStreamHandler:
    """Handles real-time audio streaming and VAD processing"""
    
    def __init__(self, vad_config: VADConfig, stream_config: Optional[AudioStreamConfig] = None):
        """Initialize audio stream handler
        
        Args:
            vad_config: VAD configuration
            stream_config: Audio stream configuration (optional)
        """
        self.vad_config = vad_config
        self.stream_config = stream_config or AudioStreamConfig()
        
        # Initialize VAD
        self.vad = VoiceActivityDetector(vad_config)
        
        # Initialize streaming components
        self._audio_queue = Queue(maxsize=self.stream_config.queue_size)
        self._stop_event = Event()
        self._stream: Optional[sd.InputStream] = None
        self._processing_thread: Optional[Thread] = None
        
        # Callback lists
        self._stream_callbacks: List[Callable[[np.ndarray], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []
        
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Callback for audio stream processing"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        try:
            # Extract audio data and convert if necessary
            audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
            if audio_chunk.dtype != self.stream_config.dtype:
                audio_chunk = audio_chunk.astype(self.stream_config.dtype)
            
            # Try to add to queue, drop if full
            try:
                self._audio_queue.put_nowait(audio_chunk)
            except Full:
                logger.warning("Audio queue full, dropping chunk")
                
            # Notify stream callbacks
            for callback in self._stream_callbacks:
                try:
                    callback(audio_chunk)
                except Exception as e:
                    logger.error(f"Error in stream callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            self._handle_error(e)
            
    def _process_audio_queue(self) -> None:
        """Process audio chunks from the queue"""
        while not self._stop_event.is_set():
            try:
                # Get audio chunk from queue with timeout
                audio_chunk = self._audio_queue.get(timeout=0.1)
                
                # Process through VAD
                results = self.vad.process_audio(audio_chunk)
                
                # Log results for debugging
                for result in results:
                    if result.is_speech:
                        logger.debug(f"Speech detected: prob={result.probability:.2f}")
                        
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio queue: {e}")
                self._handle_error(e)
                
    def start(self) -> None:
        """Start audio streaming and processing"""
        if self._stream is not None:
            logger.warning("Stream already running")
            return
            
        try:
            logger.info("Starting audio stream...")
            
            # Create input stream
            self._stream = sd.InputStream(
                device=self.stream_config.device_name,
                channels=self.stream_config.input_channels,
                samplerate=self.vad_config.sample_rate,
                blocksize=self.stream_config.buffer_size,
                dtype=self.stream_config.dtype,
                callback=self._audio_callback
            )
            
            # Clear stop event
            self._stop_event.clear()
            
            # Start processing thread
            self._processing_thread = Thread(
                target=self._process_audio_queue,
                name="AudioProcessingThread"
            )
            self._processing_thread.daemon = True
            self._processing_thread.start()
            
            # Start stream
            self._stream.start()
            logger.info("Audio stream started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self._handle_error(e)
            self.stop()
            raise
            
    def stop(self) -> None:
        """Stop audio streaming and processing"""
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
                
        # Wait for processing thread
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=2.0)
            self._processing_thread = None
            
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except Empty:
                break
                
        logger.info("Audio stream stopped")
        
    def add_stream_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Add callback for raw audio stream data"""
        self._stream_callbacks.append(callback)
        
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add callback for error handling"""
        self._error_callbacks.append(callback)
        
    def _handle_error(self, error: Exception) -> None:
        """Handle errors by notifying callbacks"""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
                
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