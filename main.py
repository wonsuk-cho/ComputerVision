# main.py
import sys
from PyQt5.QtWidgets import QApplication
from model import Model
from view import View
from controller import Controller

if __name__ == "__main__":
    app = QApplication(sys.argv)
    model = Model(min_freq=0, max_freq=100)
    view = View()
    controller = Controller(model, view)
    view.show_main_menu()
    view.show()
    sys.exit(app.exec_())
