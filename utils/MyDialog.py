from PyQt5 import QtWidgets, QtCore, QtGui

class MyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(MyDialog, self).__init__(parent, QtCore.Qt.CustomizeWindowHint)
        width = 150
        height = 50
        self.setFixedWidth(width)
        self.setFixedHeight(height)
        x = self.parent().x() + (self.parent().width() // 2) - width // 2
        y = self.parent().y() + 25
        self.move(x, y)

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        label = QtWidgets.QLabel("Please wait...")
        layout.addWidget(label, 0, 0)

        self.fontDB = QtGui.QFontDatabase()
        self.f_id = self.fontDB.addApplicationFont("resources/Seravek.ttf")

        font = QtGui.QFont('Seravek')
        font.setPointSize(16)
        label.setFont(font)