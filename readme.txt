## Project: Body Movement Sound App

### Description
The Body Movement Sound App uses a webcam to track body movements and generates sounds based on gestures. The app supports multiple modes, including:
- Normal Mode: Generates tones based on hand openness.
- Custom Mode: Plays uploaded audio with pitch changes based on movements.
- Air Drum Mode: Simulates playing drums.
- Air Piano Mode: Simulates playing piano notes.

### Prerequisites
1. Python Version: Ensure Python 3.8 or higher is installed.
2. Required Libraries: Install the following Python libraries:
   - PyQt5
   - OpenCV (cv2)
   - MediaPipe
   - NumPy
   - PyAudio
   - SimpleAudio
   - Pydub
   - PyQtGraph
   - Librosa
3. Audio Tools:
   - Install FFmpeg for `pydub` to handle audio conversions.
   - Ensure audio input/output devices are properly configured.

### Installation
Run this in your OS terminal
"pip install PyQt5 opencv-python mediapipe numpy pyaudio simpleaudio pydub pyqtgraph librosa"

### Running the Program
Just simply run the "main.py" and it should work fine.