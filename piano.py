# piano.py
import os
import numpy as np
import pyaudio
import threading
from pydub import AudioSegment
import simpleaudio as sa
from PyQt5.QtCore import Qt
from view import View

class PianoController:
    def __init__(self, view, model, sound_dir="sound"):
        self.view = view
        self.model = model
        self.sound_dir = sound_dir
        self.left_hand_notes = [
            (10, 30, "do_lower.m4a"),
            (30, 50, "re.m4a"),
            (50, 70, "mi.m4a"),
            (70, 100, "fa.m4a")
        ]
        self.right_hand_notes = [
            (10, 30, "sol.m4a"),
            (30, 50, "la.m4a"),
            (50, 70, "ti.m4a"),
            (70, 100, "do_higher.m4a")
        ]
        self.piano_sounds = self.load_piano_sounds()
        self.audio_thread = None
        self.keep_playing = False
        self.current_left_note = None
        self.current_right_note = None
        self.volume = self.model.volume
        self.lock = threading.Lock()

    def load_piano_sounds(self):
        notes = [note[2] for note in self.left_hand_notes + self.right_hand_notes]
        piano_sounds = {}
        for note in notes:
            filepath = os.path.join(self.sound_dir, note)
            if not os.path.exists(filepath):
                print(f"Piano sound file not found: {filepath}")
                piano_sounds[note] = None
                continue
            try:
                sound = AudioSegment.from_file(filepath, format="m4a")
                wav_data = sound.export(format="wav")
                wave_obj = sa.WaveObject.from_wave_file(wav_data)
                piano_sounds[note] = wave_obj
                print(f"Loaded piano sound: {filepath}")
            except Exception as e:
                print(f"Error loading piano sound ({filepath}): {e}")
                piano_sounds[note] = None
        return piano_sounds

    def map_openness_to_note(self, openness, hand='left'):
        if hand == 'left':
            notes = self.left_hand_notes
        else:
            notes = self.right_hand_notes
        for lower, upper, note in notes:
            if lower <= openness < upper:
                return note
        return notes[-1][2]

    def play_note(self, note):
        wave_obj = self.piano_sounds.get(note)
        if wave_obj:
            play_obj = wave_obj.play()
        else:
            print(f"Piano sound for {note} not loaded properly.")

    def update_notes(self, left_openness, right_openness):
        left_note = self.map_openness_to_note(left_openness, hand='left')
        right_note = self.map_openness_to_note(right_openness, hand='right')
        if left_note != self.current_left_note:
            if self.current_left_note:
                pass
            self.play_note(left_note)
            self.current_left_note = left_note
            self.view.update_current_note("Left Hand", left_note.replace('.m4a', '').capitalize())
        if right_note != self.current_right_note:
            if self.current_right_note:
                pass
            self.play_note(right_note)
            self.current_right_note = right_note
            self.view.update_current_note("Right Hand", right_note.replace('.m4a', '').capitalize())

    def start_piano(self):
        self.keep_playing = True
        self.audio_thread = threading.Thread(target=self.piano_loop, daemon=True)
        self.audio_thread.start()

    def stop_piano(self):
        self.keep_playing = False
        if self.audio_thread is not None:
            self.audio_thread.join()
            self.audio_thread = None

    def piano_loop(self):
        while self.keep_playing:
            with self.lock:
                left_openness = self.model.body_parts["left_arm"]["openness"]
                right_openness = self.model.body_parts["right_arm"]["openness"]
            self.update_notes(left_openness, right_openness)
            self.view.update_body_info(self.model.body_parts)
            self.view.update_current_note_display()
            threading.Event().wait(0.1)

    def reset_notes(self):
        self.current_left_note = None
        self.current_right_note = None
        self.view.clear_current_notes()
