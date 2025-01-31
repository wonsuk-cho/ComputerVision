# controller.py
import sys
import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import threading
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog
from view import View
from model import Model
from drum import DrumController
from piano import PianoController
import librosa
from pydub import AudioSegment
import simpleaudio as sa
import os
import time


class ReverbEffect:
    def __init__(self, sample_rate, max_delay=0.5):
        self.sample_rate = sample_rate
        self.max_delay = max_delay
        self.delay_samples = int(sample_rate * 0.3)
        self.buffer = np.zeros(self.delay_samples, dtype=np.float32)
        self.buffer_index = 0
        self.feedback = 0.5

    def apply_reverb(self, input_samples, reverb_level):
        output_samples = np.copy(input_samples)
        for i in range(len(input_samples)):
            delayed_sample = self.buffer[self.buffer_index]
            output_samples[i] += delayed_sample * reverb_level
            self.buffer[self.buffer_index] = input_samples[i] + delayed_sample * self.feedback
            self.buffer_index = (self.buffer_index + 1) % self.delay_samples
        return output_samples


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.set_main_menu_callbacks(
            self.start_normal_mode,
            self.start_custom_mode,
            self.start_air_drum_mode,
            self.start_air_piano_mode,
            self.exit_app
        )
        self.view.set_camera_callbacks(
            self.stop_camera_and_back,
            self.exit_app,
            self.set_baseline
        )
        self.view.camera_widget.volume_slider.valueChanged.connect(self.on_volume_changed)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.audio_thread = None
        self.keep_playing = False
        self.baseline_left_wrist_y = None
        self.baseline_right_wrist_y = None
        self.current_landmarks = None
        self.mode = "normal"
        self.loaded_audio_data = None
        self.loaded_sample_rate = None
        self.audio_pos = 0
        self.chunk_size = 4410
        self.alpha_openness = 0.3
        self.smoothed_left_wrist_openness = 0.0
        self.smoothed_right_wrist_openness = 0.0
        self.smoothed_left_elbow_openness = 0.0
        self.smoothed_right_elbow_openness = 0.0
        self.spectrum_frequencies = np.fft.rfftfreq(self.chunk_size, d=1. / 44100)
        self.spectrum_amplitudes = np.zeros(len(self.spectrum_frequencies))
        self.active_parts = set()
        self.reverb = ReverbEffect(sample_rate=44100)
        self.bomb_sound = self.load_bomb_sound("sound/bomb.m4a")
        self.bomb_triggered = False
        self.drum_controller = DrumController(view=self.view, model=self.model, sound_dir="sound")
        self.piano_controller = None

    def load_bomb_sound(self, filepath):
        if not os.path.exists(filepath):
            print(f"Bomb sound file not found at: {filepath}")
            return None
        try:
            sound = AudioSegment.from_file(filepath, format="m4a")
            wav_data = sound.export(format="wav")
            wave_obj = sa.WaveObject.from_wave_file(wav_data)
            return wave_obj
        except Exception as e:
            print(f"Error loading bomb sound: {e}")
            return None

    def on_volume_changed(self, value):
        self.model.set_volume_from_slider(value)

    def set_baseline(self):
        if self.current_landmarks is not None and len(self.current_landmarks) > 16:
            self.baseline_left_wrist_y = self.current_landmarks[15].y
            self.baseline_right_wrist_y = self.current_landmarks[16].y
            print(f"Baseline set: Left Wrist Y = {self.baseline_left_wrist_y}, Right Wrist Y = {self.baseline_right_wrist_y}")
        else:
            print("Insufficient landmarks to set baseline.")

    def start_normal_mode(self):
        self.mode = "normal"
        self.stop_piano_controller()
        self.stop_audio()
        self.start_camera()

    def start_custom_mode(self):
        self.mode = "custom"
        self.stop_piano_controller()
        self.stop_audio()
        file_dialog = QFileDialog()
        mp3_file, _ = file_dialog.getOpenFileName(None, "Select MP3 File", "", "MP3 Files (*.mp3)")
        if mp3_file:
            self.load_mp3_audio(mp3_file)
            self.start_camera()
        else:
            self.view.show_main_menu()

    def start_air_drum_mode(self):
        self.mode = "air_drum"
        self.stop_piano_controller()
        self.start_camera()

    def start_air_piano_mode(self):
        self.mode = "air_piano"
        self.stop_audio()
        self.start_camera()
        self.initialize_piano_controller()

    def initialize_piano_controller(self):
        if self.piano_controller is None:
            self.piano_controller = PianoController(view=self.view, model=self.model, sound_dir="sound")
        self.piano_controller.start_piano()

    def stop_piano_controller(self):
        if self.piano_controller:
            self.piano_controller.stop_piano()
            self.piano_controller = None

    def load_mp3_audio(self, filepath):
        try:
            audio = AudioSegment.from_file(filepath, format="mp3")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
                samples = samples.mean(axis=1).astype(np.float32)
            divisor = float(1 << (audio.sample_width * 8 - 1))
            samples = samples / divisor
            self.loaded_audio_data = samples
            self.loaded_sample_rate = audio.frame_rate
            self.audio_pos = 0
            print(f"Loaded custom MP3 audio: {filepath}")
        except Exception as e:
            print(f"Error loading MP3 audio: {e}")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open camera.")
            self.view.show_main_menu()
            return
        self.view.show_camera_feed()
        self.timer.start(30)
        if self.mode in ["normal", "custom"]:
            self.start_audio()

    def stop_camera_and_back(self):
        self.stop_audio()
        self.stop_piano_controller()
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.view.show_main_menu()

    def exit_app(self):
        self.stop_audio()
        self.stop_piano_controller()
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        sys.exit(0)

    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                return
            frame = cv2.flip(frame, 1)
            self.process_pose(frame)
            self.view.update_camera_frame(frame)
            self.view.update_body_info(self.model.body_parts)
            self.view.update_spectrum(self.spectrum_frequencies, self.spectrum_amplitudes)

    def process_pose(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.current_landmarks = landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            if self.mode == "air_drum":
                self.drum_controller.update(frame, landmarks)
            elif self.mode == "air_piano":
                if self.piano_controller:
                    left_openness = self.compute_openness(
                        joint_y=landmarks[15].y,
                        reference_y=self.baseline_left_wrist_y,
                        landmarks=landmarks,
                        is_left=True,
                        is_elbow=False
                    )
                    right_openness = self.compute_openness(
                        joint_y=landmarks[16].y,
                        reference_y=self.baseline_right_wrist_y,
                        landmarks=landmarks,
                        is_left=False,
                        is_elbow=False
                    )
                    self.model.update_wrist_openness(left_openness, right_openness)
                    self.piano_controller.update_notes(left_openness, right_openness)
            elif self.mode in ["normal", "custom"]:
                self.active_parts.clear()
                if self.baseline_left_wrist_y is not None and self.baseline_right_wrist_y is not None:
                    left_wrist_openness = self.compute_openness(
                        joint_y=landmarks[15].y,
                        reference_y=self.baseline_left_wrist_y,
                        landmarks=landmarks,
                        is_left=True,
                        is_elbow=False
                    )
                    right_wrist_openness = self.compute_openness(
                        joint_y=landmarks[16].y,
                        reference_y=self.baseline_right_wrist_y,
                        landmarks=landmarks,
                        is_left=False,
                        is_elbow=False
                    )
                    self.model.update_wrist_openness(left_wrist_openness, right_wrist_openness)
                    if left_wrist_openness > 50:
                        self.active_parts.add('left_arm')
                    if right_wrist_openness > 50:
                        self.active_parts.add('right_arm')
                    left_shoulder_y = landmarks[11].y
                    left_elbow_y = landmarks[13].y
                    if left_elbow_y < left_shoulder_y:
                        left_elbow_openness = self.compute_openness(
                            joint_y=left_elbow_y,
                            reference_y=left_shoulder_y,
                            landmarks=landmarks,
                            is_left=True,
                            is_elbow=True
                        )
                    else:
                        left_elbow_openness = 0.0
                    right_shoulder_y = landmarks[12].y
                    right_elbow_y = landmarks[14].y
                    if right_elbow_y < right_shoulder_y:
                        right_elbow_openness = self.compute_openness(
                            joint_y=right_elbow_y,
                            reference_y=right_shoulder_y,
                            landmarks=landmarks,
                            is_left=False,
                            is_elbow=True
                        )
                    else:
                        right_elbow_openness = 0.0
                    self.model.update_elbow_openness(left_elbow_openness, right_elbow_openness)
                    self.detect_bomb_gesture(landmarks)
                else:
                    left_wrist_openness = 0.0
                    right_wrist_openness = 0.0
                    left_elbow_openness = 0.0
                    right_elbow_openness = 0.0
                    self.model.update_wrist_openness(left_wrist_openness, right_wrist_openness)
                    self.model.update_elbow_openness(left_elbow_openness, right_elbow_openness)
                    self.view.update_smooth_values(average_openness=None, average_reverb=None)
            self.highlight_skeleton(frame, landmarks)
        else:
            if self.mode in ["normal", "custom"]:
                self.model.update_wrist_openness(0, 0)
                self.model.update_elbow_openness(0, 0)
            self.current_landmarks = None
            self.view.update_smooth_values(average_openness=None, average_reverb=None)

    def compute_openness(self, joint_y, reference_y, landmarks, is_left=True, is_elbow=False):
        if is_elbow:
            if joint_y >= reference_y:
                return 0.0
            else:
                distance = reference_y - joint_y
                max_distance = 0.2
                distance = min(distance, max_distance)
                reverb_level = distance / max_distance
                return reverb_level * 100.0
        else:
            if reference_y == 0:
                return 0.0
            if is_left:
                shoulder_y = landmarks[11].y
            else:
                shoulder_y = landmarks[12].y
            joint_y_clamped = np.clip(joint_y, shoulder_y, reference_y)
            openness = ((reference_y - joint_y_clamped) / (reference_y - shoulder_y)) * 100.0
            openness = max(0.0, min(100.0, openness))
            if is_left:
                self.smoothed_left_wrist_openness = (self.alpha_openness * openness +
                                                    (1 - self.alpha_openness) * self.smoothed_left_wrist_openness)
                return self.smoothed_left_wrist_openness
            else:
                self.smoothed_right_wrist_openness = (self.alpha_openness * openness +
                                                     (1 - self.alpha_openness) * self.smoothed_right_wrist_openness)
                return self.smoothed_right_wrist_openness

    def detect_bomb_gesture(self, landmarks):
        nose_y = landmarks[0].y
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        wrists_above_head = left_wrist.y < nose_y and right_wrist.y < nose_y
        wrist_distance = np.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)
        distance_threshold = 0.1
        wrists_close = wrist_distance < distance_threshold
        if wrists_above_head and wrists_close:
            if not self.bomb_triggered:
                self.play_bomb_sound()
                self.bomb_triggered = True
        else:
            self.bomb_triggered = False

    def play_bomb_sound(self):
        if self.bomb_sound is not None:
            self.bomb_sound.play()
        else:
            print("Bomb sound not loaded properly.")

    def highlight_skeleton(self, frame, landmarks):
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]
        nose = landmarks[0]
        left_eye = landmarks[1]
        right_eye = landmarks[2]
        left_ear = landmarks[3]
        right_ear = landmarks[4]
        connections = [
            (left_shoulder, left_elbow),
            (left_elbow, left_wrist),
            (right_shoulder, right_elbow),
            (right_elbow, right_wrist),
            (nose, left_eye),
            (nose, right_eye),
            (left_eye, left_ear),
            (right_eye, right_ear),
        ]
        for connection in connections:
            start, end = connection
            start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
            end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
            color = (0, 255, 0)
            thickness = 2
            if ((connection in [(left_shoulder, left_elbow), (left_elbow, left_wrist)] and 'left_arm' in self.active_parts) or
                (connection in [(right_shoulder, right_elbow), (right_elbow, right_wrist)] and 'right_arm' in self.active_parts)):
                color = (0, 0, 255)
                thickness = 4
            if connection in [(nose, left_eye), (nose, right_eye), (left_eye, left_ear), (right_eye, right_ear)]:
                color = (255, 0, 0)
                thickness = 2
            cv2.line(frame, start_point, end_point, color, thickness)

    def start_audio(self):
        self.keep_playing = True
        self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.audio_thread.start()

    def stop_audio(self):
        self.keep_playing = False
        if self.audio_thread is not None:
            self.audio_thread.join()
            self.audio_thread = None

    def audio_loop(self):
        p = pyaudio.PyAudio()
        output_sample_rate = 44100
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=output_sample_rate,
            output=True,
            input=False
        )
        while self.keep_playing:
            freq_left = self.model.body_parts["left_arm"]["freq_contrib"]
            freq_right = self.model.body_parts["right_arm"]["freq_contrib"]
            volume = self.model.volume
            reverb_level = self.model.current_reverb
            if self.mode == "normal":
                freq = (freq_left + freq_right) / 2.0
                if freq <= 0:
                    samples = np.zeros(int(output_sample_rate / 10), dtype=np.float32)
                else:
                    tone_t = np.linspace(0, 0.1, int(output_sample_rate * 0.1), False)
                    tone = np.sin(freq * tone_t * 2 * np.pi).astype(np.float32)
                    samples = volume * tone
            elif self.mode == "custom":
                if self.loaded_audio_data is None:
                    samples = np.zeros(int(output_sample_rate / 10), dtype=np.float32)
                else:
                    end_pos = self.audio_pos + self.chunk_size
                    if end_pos > len(self.loaded_audio_data):
                        self.audio_pos = 0
                        end_pos = self.chunk_size
                    raw_chunk = self.loaded_audio_data[self.audio_pos:end_pos]
                    self.audio_pos = end_pos
                    if freq_left <= 0 and freq_right <= 0:
                        n_steps = 0.0
                    else:
                        freq = (freq_left + freq_right) / 2.0
                        n_steps = (freq / self.model.max_freq) * 12.0
                    pitched_chunk = librosa.effects.pitch_shift(raw_chunk, sr=self.loaded_sample_rate, n_steps=n_steps)
                    desired_length = int(output_sample_rate * 0.1)
                    if len(pitched_chunk) < desired_length:
                        pad_length = desired_length - len(pitched_chunk)
                        pitched_chunk = np.pad(pitched_chunk, (0, pad_length), 'constant')
                    elif len(pitched_chunk) > desired_length:
                        pitched_chunk = pitched_chunk[:desired_length]
                    samples = (volume * pitched_chunk).astype(np.float32)
                    fft_result = np.fft.rfft(samples)
                    amplitudes = np.abs(fft_result)
                    normalized_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) != 0 else amplitudes
                    self.spectrum_amplitudes = normalized_amplitudes
            elif self.mode == "air_drum":
                samples = np.zeros(int(output_sample_rate / 10), dtype=np.float32)
            else:
                samples = np.zeros(int(output_sample_rate / 10), dtype=np.float32)
            samples = self.reverb.apply_reverb(samples, reverb_level)
            stream.write(samples.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()

    def compute_openness(self, joint_y, reference_y, landmarks, is_left=True, is_elbow=False):
        if is_elbow:
            if joint_y >= reference_y:
                return 0.0
            else:
                distance = reference_y - joint_y
                max_distance = 0.2
                distance = min(distance, max_distance)
                reverb_level = distance / max_distance
                return reverb_level * 100.0
        else:
            if reference_y == 0:
                return 0.0
            if is_left:
                shoulder_y = landmarks[11].y
            else:
                shoulder_y = landmarks[12].y
            joint_y_clamped = np.clip(joint_y, shoulder_y, reference_y)
            openness = ((reference_y - joint_y_clamped) / (reference_y - shoulder_y)) * 100.0
            openness = max(0.0, min(100.0, openness))
            if is_left:
                self.smoothed_left_wrist_openness = (self.alpha_openness * openness +
                                                    (1 - self.alpha_openness) * self.smoothed_left_wrist_openness)
                return self.smoothed_left_wrist_openness
            else:
                self.smoothed_right_wrist_openness = (self.alpha_openness * openness +
                                                     (1 - self.alpha_openness) * self.smoothed_right_wrist_openness)
                return self.smoothed_right_wrist_openness

    def detect_bomb_gesture(self, landmarks):
        nose_y = landmarks[0].y
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        wrists_above_head = left_wrist.y < nose_y and right_wrist.y < nose_y
        wrist_distance = np.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)
        distance_threshold = 0.1
        wrists_close = wrist_distance < distance_threshold
        if wrists_above_head and wrists_close:
            if not self.bomb_triggered:
                self.play_bomb_sound()
                self.bomb_triggered = True
        else:
            self.bomb_triggered = False

    def play_bomb_sound(self):
        if self.bomb_sound is not None:
            self.bomb_sound.play()
        else:
            print("Bomb sound not loaded properly.")

    def highlight_skeleton(self, frame, landmarks):
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]
        nose = landmarks[0]
        left_eye = landmarks[1]
        right_eye = landmarks[2]
        left_ear = landmarks[3]
        right_ear = landmarks[4]
        connections = [
            (left_shoulder, left_elbow),
            (left_elbow, left_wrist),
            (right_shoulder, right_elbow),
            (right_elbow, right_wrist),
            (nose, left_eye),
            (nose, right_eye),
            (left_eye, left_ear),
            (right_eye, right_ear),
        ]
        for connection in connections:
            start, end = connection
            start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
            end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
            color = (0, 255, 0)
            thickness = 2
            if ((connection in [(left_shoulder, left_elbow), (left_elbow, left_wrist)] and 'left_arm' in self.active_parts) or
                (connection in [(right_shoulder, right_elbow), (right_elbow, right_wrist)] and 'right_arm' in self.active_parts)):
                color = (0, 0, 255)
                thickness = 4
            if connection in [(nose, left_eye), (nose, right_eye), (left_eye, left_ear), (right_eye, right_ear)]:
                color = (255, 0, 0)
                thickness = 2
            cv2.line(frame, start_point, end_point, color, thickness)

    def start_piano_controller(self):
        if self.piano_controller is None:
            self.piano_controller = PianoController(view=self.view, model=self.model, sound_dir="sound")
        self.piano_controller.start_piano()

    def stop_piano_controller(self):
        if self.piano_controller:
            self.piano_controller.stop_piano()
            self.piano_controller = None
