from dataclasses import dataclass
import numpy as np
import torch
from typing import Optional, Tuple, List, Callable
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

@dataclass
class VADConfig:
    """Configuration for Voice Activity Detector"""
    sample_rate: int = 16000
    window_size: int = 512
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    device: str = 'cpu'
    model_path: Optional[Path] = None

@dataclass
class VADResult:
    """Result from Voice Activity Detection"""
    is_speech: bool
    probability: float
    timestamp: float
    audio_data: np.ndarray

class VoiceActivityDetector:
    """Production-grade Voice Activity Detector using Silero VAD"""
    
    def __init__(self, config: VADConfig):
        """Initialize VAD with configuration"""
        self.config = config
        self._initialize_model()
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_speech_timestamp: Optional[float] = None
        self._last_silence_timestamp: Optional[float] = None
        self._on_speech_callbacks: List[Callable[[VADResult], None]] = []
        self._on_silence_callbacks: List[Callable[[VADResult], None]] = []

    def _initialize_model(self):
        """Initialize Silero VAD model"""
        try:
            if self.config.model_path and self.config.model_path.exists():
                self.model = torch.jit.load(self.config.model_path)
            else:
                logger.info("Downloading Silero VAD model...")
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                self.model = model
                
            self.model.to(self.config.device)
            logger.info("VAD model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VAD model: {e}")
            raise

    def process_audio(self, audio_chunk: np.ndarray) -> List[VADResult]:
        """Process audio chunk and return VAD results
        
        Args:
            audio_chunk: numpy array of audio samples (float32)
            
        Returns:
            List of VADResult objects
        """
        results = []
        try:
            # Add new audio to buffer
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])
            
            # Process complete windows
            while len(self._audio_buffer) >= self.config.window_size:
                result = self._process_window()
                if result:
                    results.append(result)
                    
                    # Trigger callbacks
                    if result.is_speech:
                        for callback in self._on_speech_callbacks:
                            callback(result)
                    else:
                        for callback in self._on_silence_callbacks:
                            callback(result)
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            
        return results

    def _process_window(self) -> Optional[VADResult]:
        """Process one window of audio"""
        try:
            # Extract window
            window = self._audio_buffer[:self.config.window_size]
            self._audio_buffer = self._audio_buffer[self.config.window_size:]
            
            # Convert to tensor
            tensor = torch.from_numpy(window).to(self.config.device).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                speech_prob = self.model(tensor, self.config.sample_rate).item()
            
            # Apply threshold
            is_speech = speech_prob > self.config.threshold
            current_time = time.time()
            
            # Apply duration constraints
            if is_speech:
                if self._last_speech_timestamp is None:
                    self._last_speech_timestamp = current_time
                    is_speech = False  # Wait for min duration
                elif (current_time - self._last_speech_timestamp) * 1000 < self.config.min_speech_duration_ms:
                    is_speech = False  # Not long enough yet
                self._last_silence_timestamp = None
            else:
                if self._last_silence_timestamp is None:
                    self._last_silence_timestamp = current_time
                    is_speech = True  # Keep as speech during initial silence
                elif (current_time - self._last_silence_timestamp) * 1000 < self.config.min_silence_duration_ms:
                    is_speech = True  # Not silent long enough
                self._last_speech_timestamp = None
            
            return VADResult(
                is_speech=is_speech,
                probability=speech_prob,
                timestamp=current_time,
                audio_data=window if is_speech else np.array([])
            )
            
        except Exception as e:
            logger.error(f"Error processing window: {e}")
            return None

    def add_speech_callback(self, callback: Callable[[VADResult], None]):
        """Add callback for speech detection"""
        self._on_speech_callbacks.append(callback)
        
    def add_silence_callback(self, callback: Callable[[VADResult], None]):
        """Add callback for silence detection"""
        self._on_silence_callbacks.append(callback)

    def reset(self):
        """Reset internal state"""
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_speech_timestamp = None
        self._last_silence_timestamp = None