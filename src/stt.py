import numpy as np
import torch
from typing import Optional, List, Callable
from dataclasses import dataclass
from faster_whisper import WhisperModel
import logging
from vad.detector import VADConfig, VoiceActivityDetector, VADResult
import time
from queue import Queue, Empty
import sys

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription"""
    text: str
    start_time: float
    end_time: float
    confidence: float

class AudioLevelMonitor:
    """Monitor audio levels for visual feedback"""
    def __init__(self):
        self.last_update = time.time()
        self.update_interval = 0.1  # Update every 100ms
        
    def show_level(self, audio_data: np.ndarray):
        if len(audio_data) == 0:
            return
            
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            # Safely calculate audio level
            abs_data = np.abs(audio_data)
            if len(abs_data) > 0:
                level = float(np.mean(abs_data))
                if np.isfinite(level):  # Check for valid number
                    bars = int(min(max(level * 50, 0), 50))  # Clamp between 0 and 50
                    sys.stdout.write('\r')
                    sys.stdout.write(f"Level: [{('#' * bars).ljust(50)}] {level:.3f}")
                    sys.stdout.flush()
                    self.last_update = current_time

class AudioProcessor:
    """Handle audio preprocessing"""
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize audio and handle edge cases"""
        if len(audio) == 0:
            return audio
            
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
            
        # Ensure no NaN or Inf values
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return audio

class WhisperTranscriber:
    def __init__(self, model_size: str = "base"):
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8",
            download_root=None
        )
        self.current_segment: List[np.ndarray] = []
        self.is_recording = False
        self._on_transcription_callbacks: List[Callable[[TranscriptionResult], None]] = []
        self.min_audio_length = 480  # Minimum samples for processing
        logger.info("Whisper model loaded successfully")
        
    def on_speech_detected(self, audio_data: np.ndarray, timestamp: float):
        if len(audio_data) == 0:
            return
            
        if not self.is_recording:
            logger.debug("Starting new transcription segment")
            self.current_segment = []
            self.is_recording = True
            self.segment_start_time = timestamp
            
        # Preprocess audio before adding to segment
        processed_audio = AudioProcessor.normalize_audio(audio_data)
        if len(processed_audio) > 0:
            self.current_segment.append(processed_audio)
        
    def on_silence_detected(self, timestamp: float):
        if self.is_recording and self.current_segment:
            try:
                # Combine and process all audio chunks
                complete_audio = np.concatenate(self.current_segment)
                complete_audio = AudioProcessor.normalize_audio(complete_audio)
                
                if len(complete_audio) >= self.min_audio_length:
                    logger.debug(f"Processing audio segment of length: {len(complete_audio)}")
                    result = self._transcribe_audio(complete_audio, self.segment_start_time, timestamp)
                    if result and result.text.strip():
                        logger.info(f"Got transcription: {result.text}")
                        for callback in self._on_transcription_callbacks:
                            callback(result)
            except Exception as e:
                logger.error(f"Error transcribing audio: {e}")
            finally:
                self.is_recording = False
                self.current_segment = []
                
    def _transcribe_audio(self, audio: np.ndarray, start_time: float, end_time: float) -> Optional[TranscriptionResult]:
        try:
            segments, info = self.model.transcribe(
                audio,
                beam_size=3,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    threshold=0.25,
                    min_speech_duration_ms=100
                )
            )
            
            segments_list = list(segments)
            if segments_list:
                text = " ".join([seg.text for seg in segments_list])
                if text.strip():
                    avg_confidence = sum(seg.avg_logprob for seg in segments_list) / len(segments_list)
                    return TranscriptionResult(
                        text=text.strip(),
                        start_time=start_time,
                        end_time=end_time,
                        confidence=np.exp(avg_confidence)
                    )
            return None
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def add_callback(self, callback: Callable[[TranscriptionResult], None]):
        """Add callback for transcription results"""
        self._on_transcription_callbacks.append(callback)

class VoiceProcessor:
    def __init__(self, vad_config: VADConfig, whisper_config: dict):
        self.vad = VoiceActivityDetector(vad_config)
        self.transcriber = WhisperTranscriber(whisper_config.get("model_size", "base"))
        self.level_monitor = AudioLevelMonitor()
        
        self.vad.add_speech_callback(self._handle_speech)
        self.vad.add_silence_callback(self._handle_silence)
        self._on_result_callbacks: List[Callable[[TranscriptionResult], None]] = []
        self._running = False
        
        # Connect transcriber to handle results
        self.transcriber.add_callback(self._handle_transcription)
        
    def _handle_speech(self, result: VADResult):
        if len(result.audio_data) > 0:
            processed_audio = AudioProcessor.normalize_audio(result.audio_data)
            self.transcriber.on_speech_detected(processed_audio, result.timestamp)
            self.level_monitor.show_level(processed_audio)
        
    def _handle_silence(self, result: VADResult):
        self.transcriber.on_silence_detected(result.timestamp)
        if len(result.audio_data) > 0:
            self.level_monitor.show_level(result.audio_data)
        
    def _handle_transcription(self, result: TranscriptionResult):
        for callback in self._on_result_callbacks:
            callback(result)
        
    def add_result_callback(self, callback: Callable[[TranscriptionResult], None]):
        self._on_result_callbacks.append(callback)
        
    def start(self, stream_handler):
        self._running = True
        try:
            logger.info("Started processing audio...")
            print("\nListening... (Speak now)")
            print("Audio level meter below (higher is louder):")
            while self._running:
                try:
                    audio_data = stream_handler._audio_queue.get(timeout=0.1)
                    if audio_data is not None and len(audio_data) > 0:
                        # Preprocess audio before VAD
                        processed_audio = AudioProcessor.normalize_audio(audio_data)
                        self.vad.process_audio(processed_audio)
                except Empty:
                    continue
        except Exception as e:
            logger.error(f"Error processing audio stream: {e}")
            raise
            
    def stop(self):
        self._running = False

def handle_transcription(result: TranscriptionResult):
    print("\n" + "="*50)
    print(f"Transcription: {result.text}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Duration: {result.end_time - result.start_time:.1f}s")
    print("="*50)
    print("\nListening... (Audio level meter below)")

def main():
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Configure VAD with more sensitive settings
        vad_config = VADConfig(
            sample_rate=16000,
            threshold=0.25,
            min_speech_duration_ms=100,
            min_silence_duration_ms=300,
            speech_pad_ms=100,
            device='cpu'
        )
        
        # Configure Whisper
        whisper_config = {
            "model_size": "base",
        }
        
        # Initialize processor
        processor = VoiceProcessor(vad_config, whisper_config)
        processor.add_result_callback(handle_transcription)
        
        # Configure audio stream
        from vad.audio_stream import AudioStreamHandler, AudioStreamConfig
        
        stream_config = AudioStreamConfig(
            input_channels=1,
            buffer_size=1024,
            queue_size=2000,
            dtype=np.float32
        )
        
        print("\n" + "="*50)
        print("Starting transcription system...")
        print("Speak into your microphone.")
        print("You'll see an audio level meter below.")
        print("Press Ctrl+C to stop.")
        print("="*50 + "\n")
        
        with AudioStreamHandler(vad_config, stream_config) as stream:
            try:
                processor.start(stream)
            except KeyboardInterrupt:
                print("\nStopping transcription system...")
                processor.stop()
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()