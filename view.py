# view.py
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QStackedWidget, QSlider
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QSize
import pyqtgraph as pg

class MainMenuWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.start_normal_btn = QPushButton("Normal")
        self.start_custom_btn = QPushButton("Custom")
        self.start_air_drum_btn = QPushButton("Air Drum")
        self.start_air_piano_btn = QPushButton("Air Piano")
        self.exit_btn = QPushButton("Exit")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        title_label = QLabel("Body Movement Sound App")
        title_label.setFont(QFont("Arial", 24))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        self.start_normal_btn.setFont(QFont("Arial", 16))
        layout.addWidget(self.start_normal_btn)
        self.start_custom_btn.setFont(QFont("Arial", 16))
        layout.addWidget(self.start_custom_btn)
        self.start_air_drum_btn.setFont(QFont("Arial", 16))
        layout.addWidget(self.start_air_drum_btn)
        self.start_air_piano_btn.setFont(QFont("Arial", 16))
        layout.addWidget(self.start_air_piano_btn)
        self.exit_btn.setFont(QFont("Arial", 16))
        layout.addWidget(self.exit_btn)
        self.setLayout(layout)

    def set_callbacks(self, normal_callback, custom_callback, air_drum_callback, air_piano_callback, exit_callback):
        self.start_normal_btn.clicked.connect(normal_callback)
        self.start_custom_btn.clicked.connect(custom_callback)
        self.start_air_drum_btn.clicked.connect(air_drum_callback)
        self.start_air_piano_btn.clicked.connect(air_piano_callback)
        self.exit_btn.clicked.connect(exit_callback)


class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.back_btn = QPushButton("Back")
        self.exit_btn = QPushButton("Exit")
        self.camera_label = QLabel("Camera Feed")
        self.info_layout = QVBoxLayout()
        self.volume_slider = QSlider(Qt.Horizontal)
        self.set_baseline_btn = QPushButton("Calibrate Camera")
        self.smoothed_left_arm_label = QLabel("Smoothed Left Arm Openness: 0.00%, Freq: 0.00 Hz")
        self.smoothed_right_arm_label = QLabel("Smoothed Right Arm Openness: 0.00%, Freq: 0.00 Hz")
        self.smoothed_left_elbow_label = QLabel("Smoothed Left Elbow Openness: 0.00%, Reverb: 0.00")
        self.smoothed_right_elbow_label = QLabel("Smoothed Right Elbow Openness: 0.00%, Reverb: 0.00")
        self.init_ui()

    def init_ui(self):
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.setFixedSize(640, 480)
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setYRange(0, 1)
        self.spectrum_plot.setLabel('left', 'Amplitude')
        self.spectrum_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.spectrum_plot.showGrid(x=True, y=True)
        self.spectrum_plot.hideButtons()
        self.spectrum_curve = self.spectrum_plot.plot(pen='y')
        self.info_label = QLabel("Body Part States:")
        self.info_label.setFont(QFont("Arial", 14))
        self.info_layout.addWidget(self.info_label)
        self.info_layout.addWidget(self.smoothed_left_arm_label)
        self.info_layout.addWidget(self.smoothed_right_arm_label)
        self.info_layout.addWidget(self.smoothed_left_elbow_label)
        self.info_layout.addWidget(self.smoothed_right_elbow_label)
        self.current_note_label = QLabel("")
        self.current_note_label.setFont(QFont("Arial", 16))
        self.current_note_label.setAlignment(Qt.AlignCenter)
        self.info_layout.addWidget(self.current_note_label)
        volume_layout = QHBoxLayout()
        volume_label = QLabel("Volume:")
        volume_label.setFont(QFont("Arial", 12))
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(25)
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_slider)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.back_btn)
        btn_layout.addWidget(self.exit_btn)
        btn_layout.addWidget(self.set_baseline_btn)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.camera_label)
        main_layout.addWidget(self.spectrum_plot)
        main_layout.addLayout(self.info_layout)
        main_layout.addLayout(volume_layout)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def set_callbacks(self, back_callback, exit_callback, baseline_callback):
        self.back_btn.clicked.connect(back_callback)
        self.exit_btn.clicked.connect(exit_callback)
        self.set_baseline_btn.clicked.connect(baseline_callback)

    def update_current_note(self, hand, note):
        existing_text = self.current_note_label.text()
        new_text = f"{hand}: {note}"
        if existing_text:
            new_text = existing_text + " | " + new_text
        self.current_note_label.setText(new_text)

    def clear_current_notes(self):
        self.current_note_label.setText("")


class View(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Body Movement Sound App")
        self.setMinimumSize(800, 800)
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(0, 122, 204))
        self.setPalette(dark_palette)
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.main_menu = MainMenuWidget()
        self.camera_widget = CameraWidget()
        self.stacked_widget.addWidget(self.main_menu)
        self.stacked_widget.addWidget(self.camera_widget)

    def set_main_menu_callbacks(self, normal_callback, custom_callback, air_drum_callback, air_piano_callback, exit_callback):
        self.main_menu.set_callbacks(normal_callback, custom_callback, air_drum_callback, air_piano_callback, exit_callback)

    def set_camera_callbacks(self, back_callback, exit_callback, baseline_callback):
        self.camera_widget.set_callbacks(back_callback, exit_callback, baseline_callback)

    def show_main_menu(self):
        self.stacked_widget.setCurrentWidget(self.main_menu)

    def show_camera_feed(self):
        self.stacked_widget.setCurrentWidget(self.camera_widget)

    def update_camera_frame(self, frame):
        if frame is not None:
            rgb_frame = frame[..., ::-1]
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            data = rgb_frame.tobytes()
            qimg = QImage(data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            scaled_pix = pix.scaled(self.camera_widget.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_widget.camera_label.setPixmap(scaled_pix)

    def update_body_info(self, body_parts_dict):
        self.camera_widget.smoothed_left_arm_label.setText(
            f"Smoothed Right Arm Openness: {body_parts_dict['left_arm']['openness']:.2f}%, Freq: {body_parts_dict['left_arm']['freq_contrib']:.2f} Hz"
        )
        self.camera_widget.smoothed_right_arm_label.setText(
            f"Smoothed Left Arm Openness: {body_parts_dict['right_arm']['openness']:.2f}%, Freq: {body_parts_dict['right_arm']['freq_contrib']:.2f} Hz"
        )
        self.camera_widget.smoothed_left_elbow_label.setText(
            f"Smoothed Right Elbow Openness: {body_parts_dict['left_elbow']['openness']:.2f}%, Reverb: {body_parts_dict['left_elbow']['reverb_contrib']:.2f}"
        )
        self.camera_widget.smoothed_right_elbow_label.setText(
            f"Smoothed Left Elbow Openness: {body_parts_dict['right_elbow']['openness']:.2f}%, Reverb: {body_parts_dict['right_elbow']['reverb_contrib']:.2f}"
        )

    def update_spectrum(self, frequencies, amplitudes):
        self.camera_widget.spectrum_curve.setData(frequencies, amplitudes)

    def update_current_note(self, hand, note):
        existing_text = self.camera_widget.current_note_label.text()
        new_text = f"{hand}: {note}"
        if existing_text:
            new_text = existing_text + " | " + new_text
        self.camera_widget.current_note_label.setText(new_text)

    def clear_current_notes(self):
        self.camera_widget.current_note_label.setText("")
