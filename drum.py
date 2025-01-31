# drum.py
import cv2
import numpy as np
import time
import os
from pydub import AudioSegment
import simpleaudio as sa

class DrumController:
    def __init__(self, view, model, sound_dir="sound", hit_cooldown=0.5, highlight_duration=0.2):
        self.view = view
        self.model = model
        self.sound_dir = sound_dir
        self.hit_cooldown = hit_cooldown
        self.highlight_duration = highlight_duration
        self.drum_sounds = self.load_drum_sounds()
        self.drum_positions = {}
        self.circle_radius_multiplier = 0.07
        self.last_hit_time = {
            "hihat": 0,
            "snare": 0,
            "bass": 0
        }
        self.hit_highlight_times = {
            "hihat": 0,
            "snare": 0,
            "bass": 0
        }

    def load_drum_sounds(self):
        drum_names = ["hihat", "snare", "bass"]
        drum_sounds = {}
        for drum in drum_names:
            filepath = os.path.join(self.sound_dir, f"{drum}.m4a")
            if not os.path.exists(filepath):
                print(f"Drum sound file not found: {filepath}")
                drum_sounds[drum] = None
                continue
            try:
                sound = AudioSegment.from_file(filepath, format="m4a")
                wav_data = sound.export(format="wav")
                wave_obj = sa.WaveObject.from_wave_file(wav_data)
                drum_sounds[drum] = wave_obj
                print(f"Loaded drum sound: {filepath}")
            except Exception as e:
                print(f"Error loading drum sound ({filepath}): {e}")
                drum_sounds[drum] = None
        return drum_sounds

    def draw_drum_circles(self, frame):
        height, width, _ = frame.shape
        circle_radius = int(min(width, height) * self.circle_radius_multiplier)
        hihat_center = (int(width * 0.3), int(height * 0.25))
        snare_center = (int(width * 0.5), int(height * 0.25))
        bass_center = (int(width * 0.7), int(height * 0.25))
        cv2.circle(frame, hihat_center, circle_radius, (0, 255, 255), 2)
        cv2.circle(frame, snare_center, circle_radius, (0, 0, 255), 2)
        cv2.circle(frame, bass_center, circle_radius, (255, 0, 0), 2)
        cv2.putText(frame, "Hi-hat", (hihat_center[0] - 40, hihat_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Snare", (snare_center[0] - 35, snare_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Bass", (bass_center[0] - 30, bass_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        self.drum_positions = {
            "hihat": hihat_center,
            "snare": snare_center,
            "bass": bass_center
        }

    def is_point_inside_circle(self, point, center, radius):
        distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
        return distance <= radius

    def detect_drum_hits(self, frame, landmarks):
        if not self.drum_positions:
            return
        frame_height, frame_width, _ = frame.shape
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_wrist_px = (int(left_wrist.x * frame_width), int(left_wrist.y * frame_height))
        right_wrist_px = (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height))
        detection_radius = int(min(frame_width, frame_height) * self.circle_radius_multiplier)
        for drum, center in self.drum_positions.items():
            hit = False
            if self.is_point_inside_circle(left_wrist_px, center, detection_radius):
                hit = True
            if self.is_point_inside_circle(right_wrist_px, center, detection_radius):
                hit = True
            if hit:
                self.play_drum_sound(drum)
                self.hit_highlight_times[drum] = time.time()
        for drum, center in self.drum_positions.items():
            if time.time() - self.hit_highlight_times.get(drum, 0) < self.highlight_duration:
                overlay = frame.copy()
                cv2.circle(overlay, center, detection_radius, (0, 255, 0), -1)
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def play_drum_sound(self, drum):
        current_time = time.time()
        last_hit = self.last_hit_time.get(drum, 0)
        if current_time - last_hit > self.hit_cooldown:
            wave_obj = self.drum_sounds.get(drum)
            if wave_obj:
                play_obj = wave_obj.play()
                self.last_hit_time[drum] = current_time
            else:
                print(f"Drum sound for {drum} not loaded properly.")

    def update(self, frame, landmarks):
        self.draw_drum_circles(frame)
        self.detect_drum_hits(frame, landmarks)