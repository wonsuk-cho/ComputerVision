import sys
import cv2
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from pyo import Server, Sine

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize the Pyo audio server
s = Server().boot()
s.start()

# Set up initial frequency (A4 pitch)
frequency = 440  # Normal pitch
oscillator = Sine(freq=frequency).out()

# Function to update pitch based on hand state (open/closed)
def update_pitch(is_hand_open):
    global frequency
    if is_hand_open:
        frequency = 440  # Open hand -> normal pitch (A4)
    else:
        frequency = 220  # Closed hand -> lower pitch (A3)
    oscillator.setFreq(frequency)

# Function to check if hand is open or closed based on landmarks
def is_hand_open(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    return abs(thumb_tip - index_tip) > 0.1  # Threshold for open/closed hand

# PyQt5 GUI for live video and hand tracking
class HandSoundApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hand Sound App")

        # Video display label
        self.video_label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        # Timer to update video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms

        # Video capture from the webcam
        self.cap = cv2.VideoCapture(0)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB for Mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            # Check for hand landmarks and draw them
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark

                    # Check if the hand is open or closed and update the pitch
                    if is_hand_open(landmarks):
                        update_pitch(True)  # Hand open, normal pitch
                    else:
                        update_pitch(False)  # Hand closed, lower pitch

            # Convert the frame to Qt-compatible image format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Display the frame in the label
            self.video_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        # Release video capture and stop the sound when closing the app
        self.cap.release()
        s.stop()

# Main application loop
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandSoundApp()
    window.show()
    sys.exit(app.exec_())