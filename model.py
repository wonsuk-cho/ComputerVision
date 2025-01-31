# model.py
import numpy as np

class Model:
    def __init__(self, min_freq=50, max_freq=300):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.current_reverb = 0.0
        self.body_parts = {
            "left_arm": {"openness": 0.0, "freq_contrib": 0.0},
            "right_arm": {"openness": 0.0, "freq_contrib": 0.0},
            "left_elbow": {"openness": 0.0, "reverb_contrib": 0.0},
            "right_elbow": {"openness": 0.0, "reverb_contrib": 0.0}
        }
        self.volume = 0.5

    def update_wrist_openness(self, left_wrist_openness, right_wrist_openness):
        self.body_parts["left_arm"]["openness"] = left_wrist_openness
        if left_wrist_openness <= 0:
            self.body_parts["left_arm"]["freq_contrib"] = self.min_freq
        else:
            self.body_parts["left_arm"]["freq_contrib"] = self.min_freq + (self.max_freq - self.min_freq) * (left_wrist_openness / 100.0)
        self.body_parts["right_arm"]["openness"] = right_wrist_openness
        if right_wrist_openness <= 0:
            self.body_parts["right_arm"]["freq_contrib"] = self.min_freq
        else:
            self.body_parts["right_arm"]["freq_contrib"] = self.min_freq + (self.max_freq - self.min_freq) * (right_wrist_openness / 100.0)

    def update_elbow_openness(self, left_elbow_openness, right_elbow_openness):
        self.body_parts["left_elbow"]["openness"] = left_elbow_openness
        self.body_parts["left_elbow"]["reverb_contrib"] = left_elbow_openness / 100.0 if left_elbow_openness > 0 else 0.0
        self.body_parts["right_elbow"]["openness"] = right_elbow_openness
        self.body_parts["right_elbow"]["reverb_contrib"] = right_elbow_openness / 100.0 if right_elbow_openness > 0 else 0.0
        total_reverb = self.body_parts["left_elbow"]["reverb_contrib"] + self.body_parts["right_elbow"]["reverb_contrib"]
        self.current_reverb = min(total_reverb, 1.0)

    def set_volume_from_slider(self, slider_value):
        self.volume = slider_value / 100.0
