from PyQt5 import QtCore, QtWidgets

class MySlider(QtWidgets.QSlider):
    def __init__(self):
        super(MySlider, self).__init__(QtCore.Qt.Horizontal)

        # style = "QSlider::groove:horizontal { background-color: #C0C0C0; border: 0px solid  #424242; height: 5px; border-radius: 2px; } QSlider::handle:horizontal { background-color: red; border: 1px solid red; width: 8px; height: 10px; line-height: 10px; margin-top: -2.5px; margin-bottom: -2.5px; border-radius: 5px; }"
        # style = "QSlider::handle:horizontal { 	width: 16px; height: 16px; margin: -5px 6px -5px 6px; border-radius:11px; border: 3px solid #ffffff;}"
        # style = "QSlider { background-color: red; border-style: outset; border-width: 2px; border-radius: 10px; border-color: beige; }"
        # self.setStyleSheet(style)

        self.interactable = True

    def mousePressEvent(self, QMouseEvent):
        if self.interactable is False:
            return
        if QMouseEvent.button() == QtCore.Qt.LeftButton:
            if self.orientation() == QtCore.Qt.Vertical:
                self.setValue(self.minimum() + ((self.maximum() - self.minimum()) * (self.height() - QMouseEvent.y())) / self.height())
            else:
                self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), QMouseEvent.x(), self.width()))
        super(MySlider, self).mousePressEvent(QMouseEvent)

    def mouseMoveEvent(self, QMouseEvent):
        if self.interactable is False:
            return
        if self.orientation() == QtCore.Qt.Vertical:
            self.setValue(self.minimum() + (
                        (self.maximum() - self.minimum()) * (self.height() - QMouseEvent.y())) / self.height())
        else:
            self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), QMouseEvent.x(), self.width()))
        super(MySlider, self).mouseMoveEvent(QMouseEvent)

    def mouseReleaseEvent(self, QMouseEvent):
        if self.interactable is False:
            return
        if self.orientation() == QtCore.Qt.Vertical:
            self.setValue(self.minimum() + (
                        (self.maximum() - self.minimum()) * (self.height() - QMouseEvent.y())) / self.height())
        else:
            self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), QMouseEvent.x(), self.width()))
        super(MySlider, self).mouseReleaseEvent(QMouseEvent)

