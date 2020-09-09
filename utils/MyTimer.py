from PyQt5 import QtCore

class MyTimer():
    def __init__(self, parent=None, timeoutCallback=None):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.onTimeout)
        self.timeoutCallback = timeoutCallback

    # in seconds
    def start(self, time):
        self.timer.start(time * 1000)
        self.max_time = time * 1000
        self.total = self.max_time

    def pause(self):
        remain = self.timer.remainingTime()
        consume = self.max_time - remain
        self.max_time -= consume
        self.timer.stop()

    def resume(self):
        self.timer.start(self.max_time)

    def stop(self):
        self.timer.stop()

    def getTime(self):
        if self.timer.isActive():
            return (self.total - self.timer.remainingTime()) / 1000.0
        else:
            return (self.total - self.max_time) / 1000.0

    def onTimeout(self):
        self.timer.stop()
        self.timeoutCallback()
