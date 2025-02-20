import logging
import sys
from pathlib import Path
from typing import Optional
import time
import wave
import numpy as np
from vad.detector import VADConfig, VADResult
from vad.audio_stream import AudioStreamHandler, AudioStreamConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class VoiceRecorder:
    """Records voice segments when speech is detected"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.current_wave: Optional[wave.Wave_write] = None
        self.recording = False
        self.segment_count = 0
        
    def on_speech_detected(self, result: VADResult):
        """Handle speech detection"""
        if not self.recording:
            self.start_recording()
        if result.audio_data.size > 0:
            self.write_audio(result.audio_data)
            
    def on_silence_detected(self, result: VADResult):
        """Handle silence detection"""
        if self.recording:
            self.stop_recording()
            
    def start_recording(self):
        """Start a new recording segment"""
        if self.current_wave is not None:
            self.stop_recording()
            
        self.segment_count += 1
        output_path = self.output_dir / f"segment_{self.segment_count}.wav"
        
        self.current_wave = wave.open(str(output_path), 'wb')
        self.current_wave.setnchannels(1)
        self.current_wave.setsampwidth(2)  # 16-bit audio
        self.current_wave.setframerate(16000)
        self.recording = True
        logger.info(f"Started recording segment {self.segment_count}")
        
    def stop_recording(self):
        """Stop current recording segment"""
        if self.current_wave is not None:
            self.current_wave.close()
            self.current_wave = None
            self.recording = False
            logger.info(f"Finished recording segment {self.segment_count}")
            
    def write_audio(self, audio_data: np.ndarray):
        """Write audio data to current wave file"""
        if self.current_wave is not None:
            # Convert float32 to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            self.current_wave.writeframes(audio_int16.tobytes())

def main():
    """Main function demonstrating VAD usage"""
    try:
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Initialize recorder
        recorder = VoiceRecorder(output_dir)
        
        # Configure VAD
        vad_config = VADConfig(
            sample_rate=16000,
            window_size=512,
            threshold=0.5,
            min_speech_duration_ms=300,
            min_silence_duration_ms=100,
            device='cpu'
        )
        
        # Configure audio stream
        stream_config = AudioStreamConfig(
            input_channels=1,
            buffer_size=1024,
            queue_size=1000
        )
        
        # Create audio stream handler
        handler = AudioStreamHandler(vad_config, stream_config)
        
        # Add callbacks
        handler.vad.add_speech_callback(recorder.on_speech_detected)
        handler.vad.add_silence_callback(recorder.on_silence_detected)
        
        # Error callback
        def on_error(e: Exception):
            logger.error(f"Error in audio processing: {e}")
        handler.add_error_callback(on_error)
        
        # Start streaming
        logger.info("Starting VAD system. Press Ctrl+C to stop.")
        with handler:
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Stopping VAD system...")
                
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()