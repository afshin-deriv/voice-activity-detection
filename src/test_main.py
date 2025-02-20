import torch
import torchaudio
import numpy as np
import sounddevice as sd
from queue import Queue
from threading import Thread
import time

# Constants for audio processing
SAMPLE_RATE = 16000
WINDOW_SIZE = 512  # Correct window size for 16kHz sampling rate
CHANNELS = 1
DEVICE = 'cpu'  # Change to 'cuda' if you have a GPU

class VoiceActivityDetector:
    def __init__(self):
        # Download and initialize Silero VAD model
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        self.model = model.to(DEVICE)
        self.get_speech_timestamps = utils[0]
        self.audio_queue = Queue()
        self.vad_running = False

    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice to handle incoming audio"""
        if status:
            print(f"Status: {status}")
        # Convert to float32 and reshape
        audio_data = indata[:, 0].copy().astype(np.float32)
        self.audio_queue.put(audio_data)

    def process_audio(self):
        """Process audio chunks with Silero VAD"""
        buffer = np.array([], dtype=np.float32)
        
        while self.vad_running:
            if not self.audio_queue.empty():
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get()
                buffer = np.concatenate([buffer, audio_chunk])
                
                # Process when buffer has enough samples
                while len(buffer) >= WINDOW_SIZE:
                    # Extract window and convert to tensor
                    window = buffer[:WINDOW_SIZE]
                    tensor = torch.from_numpy(window).to(DEVICE)
                    
                    # Ensure tensor shape is correct (add batch dimension)
                    tensor = tensor.unsqueeze(0)
                    
                    # Run VAD inference
                    speech_prob = self.model(tensor, SAMPLE_RATE).item()
                    
                    # Print VAD result with simple visualization
                    confidence = int(speech_prob * 50)  # Scale for visualization
                    speech_indicator = "SPEECH" if speech_prob > 0.5 else "SILENCE"
                    print(f"{speech_indicator} [{('#' * confidence).ljust(50)}] {speech_prob:.2f}", end='\r')
                    
                    # Remove processed samples
                    buffer = buffer[WINDOW_SIZE:]
            
            time.sleep(0.01)  # Small sleep to prevent CPU overuse

    def start(self):
        """Start recording and VAD processing"""
        print("Starting Voice Activity Detection...")
        print("Speak into your microphone. Press Ctrl+C to stop.")
        print("Speech probability will be shown in real-time.")
        print("Values closer to 1.0 indicate speech, closer to 0.0 indicate silence.")
        
        self.vad_running = True
        
        # Start processing thread
        process_thread = Thread(target=self.process_audio)
        process_thread.start()
        
        # Start audio recording
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self.audio_callback,
            blocksize=WINDOW_SIZE
        ):
            try:
                while self.vad_running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping Voice Activity Detection...")
                self.vad_running = False
                process_thread.join()

def main():
    vad = VoiceActivityDetector()
    vad.start()

if __name__ == "__main__":
    main()
