import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from PyQt5 import QtWidgets, QtCore, QtMultimedia
from utils.utils import *
from utils.QtPlot import QtPlot
from utils.MyDialog import MyDialog
from utils.MyTimer import MyTimer

class RandomEvaluationUI(QtWidgets.QWidget):
    def __init__(self, model, output_path, data_ui=True):
        super(RandomEvaluationUI, self).__init__()

        self.model = model
        self.output_base_path = output_path
        self.data_ui = data_ui

        # self.latent_size = model.latent_size
        self.latent_size = 256

        # self.data_dim = model.data_dim
        self.data_dim = 1

        # Parameters
        self.range = -1

        # Variables (System)
        self.target_latent = None
        self.target_data = None
        self.center_latent = None
        self.center_data = None
        self.current_latent = None
        self.current_data = None
        self.current_best_data = None

        self.best_choices = []
        self.update_as_best = False

        self.current_path = None

        self.initUI()

        # Timer
        self.timer = MyTimer(self, self.OnSaveButtonPressed)

        self.show()

    def initUI(self):
        self.main_width = 600
        self.main_height = 300
        self.setGeometry(600, 515, self.main_width, self.main_height)

        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)

        if self.data_ui:
            self.data_lable_layout = QtWidgets.QGridLayout()
            label = QtWidgets.QLabel('')
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            label.setFixedHeight(12)
            self.data_lable_layout.addWidget(label, 0, 0, QtCore.Qt.AlignHCenter)

            label = QtWidgets.QLabel('Target')
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            label.setFixedHeight(12)
            self.data_lable_layout.addWidget(label, 0, 1, QtCore.Qt.AlignHCenter)

            label = QtWidgets.QLabel('Current Best')
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            self.data_lable_layout.addWidget(label, 0, 2, QtCore.Qt.AlignHCenter)
            main_layout.addLayout(self.data_lable_layout)

        button_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(button_layout)

        self.update_as_best_button = QtWidgets.QPushButton()
        self.update_as_best_button.setStyleSheet("border-image:url(resources/sad.png);")
        self.update_as_best_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.update_as_best_button.setFixedSize(42, 42)
        self.update_as_best_button.clicked.connect(self.OnUpdateAsBestButtonClicked)
        button_layout.addWidget(self.update_as_best_button)

        self.next_button = QtWidgets.QPushButton()
        self.next_button.setStyleSheet("QPushButton{border-image:url(resources/next.png);} QPushButton:pressed{border-image:url(resources/next_p.png);}")
        self.next_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.next_button.setFixedSize(85, 45)
        self.next_button.clicked.connect(self.OnNextButtonPressed)
        button_layout.addWidget(self.next_button)

    def init(self):
        self.current_updated_flag = True
        self.target_updated_flag = True
        self.current_best_updated_flag = True

    def paint(self):
        pass

    def start_evaluation(self, init_latent, target_latent, time, case_idx=None):
        self.best_choices = []
        self.target_latent = target_latent
        self.target_data = self.model.decode(self.target_latent.reshape(1, -1))[0]

        self.center_latent = init_latent
        self.center_data = self.model.decode(self.center_latent.reshape(1, -1))[0]
        self.current_data = self.center_data
        self.current_best_data = self.current_data
        self.addNewChoice(0.0, self.current_data)
        self.best_choices[-1]['iteration'] = 0

        safe_mkdir(self.output_base_path)
        if case_idx is None:
            case_idx = len(os.listdir(self.output_base_path))
        self.case_idx = case_idx

        self.range = np.sqrt(np.sum((self.target_latent - self.center_latent) ** 2))

        self.init()

        self.search_iter_count = 0

        self.timer.start(time)

        if self.update_as_best is True:
            self.OnUpdateAsBestButtonClicked()

        self.paint()

    def sample_new_point(self):
        latent = self.model.get_random_latent()
        self.current_latent = latent
        self.current_data = self.model.decode(self.current_latent.reshape(1, -1))[0]

    def save_search_result(self):
        safe_mkdir(self.output_base_path)
        self.current_output_path = self.output_base_path + '/' + str(self.case_idx)
        safe_mkdir(self.current_output_path)

        filename = self.current_output_path + '/time.txt'
        f = open(filename, 'w')
        f.write(str(len(self.best_choices)) + "\n")
        for i in range(len(self.best_choices)):
            f.write(str(self.best_choices[i]['time']) + "\n")
        f.close()

        filename = self.current_output_path + '/iteration_count.txt'
        f = open(filename, 'w')
        f.write(str(len(self.best_choices)) + "\n")
        for i in range(len(self.best_choices)):
            f.write(str(self.best_choices[i]['iteration']) + "\n")
        f.write(str(self.search_iter_count) + "\n")
        f.close()

    def addNewChoice(self, time, data):
        self.best_choices.append({'time': time, 'data': data})

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_S:
            self.OnSaveButtonPressed()

    def OnNextButtonPressed(self):
        self.search_iter_count += 1

        if self.update_as_best:
            self.current_best_updated_flag = True
            self.current_best_data = self.current_data
            self.addNewChoice(self.timer.getTime(), self.current_data)
            self.best_choices[-1]['iteration'] = self.search_iter_count
            self.OnUpdateAsBestButtonClicked()

        self.sample_new_point()
        self.paint()
        self.current_updated_flag = True

    def OnSaveButtonPressed(self):
        self.save_search_result()
        self.myClose()

    def OnUpdateAsBestButtonClicked(self):
        if self.update_as_best:
            self.update_as_best = False
            self.update_as_best_button.setStyleSheet("border-image:url(resources/sad.png);")
        else:
            self.update_as_best = True
            self.update_as_best_button.setStyleSheet("border-image:url(resources/smile.png);")

    def myShow(self):
        self.show()

    def myClose(self):
        self.timer.stop()
        self.close()

class ImageRandomEvaluationUI(RandomEvaluationUI):
    def __init__(self, model, output_path):
        super(ImageRandomEvaluationUI, self).__init__(model, output_path)

    def initUI(self):
        super(ImageRandomEvaluationUI, self).initUI()

        size = (self.main_width - 50) / 3
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.current_image_canvas = QtPlot()
        self.current_image_canvas.setSizePolicy(sizePolicy)
        self.current_image_canvas.setFixedSize(size, size)
        self.data_lable_layout.addWidget(self.current_image_canvas, 1, 0)

        self.target_image_canvas = QtPlot()
        self.target_image_canvas.setSizePolicy(sizePolicy)
        self.target_image_canvas.setFixedSize(size, size)
        self.data_lable_layout.addWidget(self.target_image_canvas, 1, 1)

        self.current_best_image_canvas = QtPlot()
        self.current_best_image_canvas.setSizePolicy(sizePolicy)
        self.current_best_image_canvas.setFixedSize(size, size)
        self.data_lable_layout.addWidget(self.current_best_image_canvas, 1, 2)

    def paint(self):
        super(ImageRandomEvaluationUI, self).paint()

        self.current_image_canvas.clear()
        self.target_image_canvas.clear()
        self.current_best_image_canvas.clear()

        cmap = 'gray'
        if self.data_dim != 2:
            cmap = None

        if self.current_data is not None:
            self.current_image_canvas.imshow(self.current_data, cmap)
            self.current_image_canvas.draw()

        if self.target_data is not None:
            self.target_image_canvas.imshow(self.target_data, cmap)
            self.target_image_canvas.draw()

        if self.current_best_data is not None:
            self.current_best_image_canvas.imshow(self.current_best_data, cmap)
            self.current_best_image_canvas.draw()

    def save_search_result(self):
        import cv2

        super(ImageRandomEvaluationUI, self).save_search_result()

        for i in range(len(self.best_choices)):
            img = self.best_choices[i]['data']
            if self.data_dim != 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            fig_name = self.current_output_path + '/' + str(i) + '.png'
            cv2.imwrite(fig_name, img)

        img = self.target_data
        if self.data_dim != 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fig_name = self.current_output_path + '/target.png'
        cv2.imwrite(fig_name, img)

class AudioRandomEvaluationUI(RandomEvaluationUI):
    def __init__(self, model, output_path, sr):
        super(AudioRandomEvaluationUI, self).__init__(model, output_path)

        # Audio Player Parameters
        self.volume_param = 20000
        self.sr = sr

        # Audio Player
        self.initAudioPlayer()

    def initUI(self):
        super(AudioRandomEvaluationUI, self).initUI()

        self.play_found_sound_button = QtWidgets.QPushButton()
        self.play_found_sound_button.setStyleSheet("QPushButton{border-image:url(resources/play_sound_square.png);}")
        self.play_found_sound_button.setFixedSize(150, 150)
        self.play_found_sound_button.clicked.connect(self.OnPlayFoundSoundButtonClicked)
        self.data_lable_layout.addWidget(self.play_found_sound_button, 1, 0, QtCore.Qt.AlignCenter)

        self.play_target_sound_button = QtWidgets.QPushButton()
        self.play_target_sound_button.setStyleSheet("QPushButton{border-image:url(resources/play_sound_square.png);}")
        self.play_target_sound_button.setFixedSize(150, 150)
        self.play_target_sound_button.clicked.connect(self.OnPlayTargetSoundButtonClicked)
        self.data_lable_layout.addWidget(self.play_target_sound_button, 1, 1, QtCore.Qt.AlignCenter)

        self.play_current_best_sound_button = QtWidgets.QPushButton()
        self.play_current_best_sound_button.setStyleSheet("QPushButton{border-image:url(resources/play_sound_square.png);}")
        self.play_current_best_sound_button.setFixedSize(150, 150)
        self.play_current_best_sound_button.clicked.connect(self.OnPlayCurrentBestSoundButtonClicked)
        self.data_lable_layout.addWidget(self.play_current_best_sound_button, 1, 2, QtCore.Qt.AlignCenter)

    def initAudioPlayer(self):
        self.qbyte_array = QtCore.QByteArray()
        self.buffer = QtCore.QBuffer()

        audioFormat = QtMultimedia.QAudioFormat()
        audioFormat.setSampleRate(self.sr)
        audioFormat.setChannelCount(1)
        audioFormat.setSampleSize(16)
        audioFormat.setCodec('audio/pcm')
        audioFormat.setByteOrder(QtMultimedia.QAudioFormat.LittleEndian)
        audioFormat.setSampleType(QtMultimedia.QAudioFormat.SignedInt)

        deviceInfo = QtMultimedia.QAudioDeviceInfo(QtMultimedia.QAudioDeviceInfo.defaultOutputDevice())
        if not deviceInfo.isFormatSupported(audioFormat):
            print('Raw audio format not supported by backend, cannot play audio.')
            return

        self.audioPlayer = QtMultimedia.QAudioOutput(audioFormat, self)

    def play_sound(self, sound):
        import struct

        self.buffer.close()

        clipped = np.clip(sound * self.volume_param, -32768, 32767)

        n = sound.shape[0]

        self.qbyte_array.clear()
        for i in range(n):
            self.qbyte_array.append(struct.pack('<h', int(clipped[i])))

        self.buffer.setData(self.qbyte_array)
        self.buffer.open(QtCore.QIODevice.ReadOnly)
        self.buffer.seek(0)

        self.audioPlayer.start(self.buffer)

    def save_search_result(self):
        super(AudioRandomEvaluationUI, self).save_search_result()

        for i in range(len(self.best_choices)):
            sound = self.best_choices[i]['data']
            sound_name = self.current_output_path + '/' + str(i) + '.wav'
            self.write_wave(sound, sound_name, self.sr)

        sound = self.target_data
        sound_name = self.current_output_path + '/target.wav'
        self.write_wave(sound, sound_name, self.sr)

    def write_wave(self, wave, name, sr):
        import librosa

        librosa.output.write_wav(name, wave.astype(np.float32), sr)

    def OnPlayFoundSoundButtonClicked(self):
        if self.current_data is None:
            return
        sound = self.current_data
        self.play_sound(sound)
    def OnPlayTargetSoundButtonClicked(self):
        if self.target_data is None:
            return
        sound = self.target_data
        self.play_sound(sound)

    def OnPlayCurrentBestSoundButtonClicked(self):
        if self.current_best_data is None:
            return
        sound = self.current_best_data
        self.play_sound(sound)

    def OnNextButtonPressed(self):
        super(AudioRandomEvaluationUI, self).OnNextButtonPressed()
        self.play_sound(self.current_data)

class OpenGLRandomEvaluationUI(RandomEvaluationUI):
    def __init__(self, model, output_path, ref_opengl_widget, ref_window):
        self.opengl_widget = ref_opengl_widget
        self.ref_window = ref_window
        super(OpenGLRandomEvaluationUI, self).__init__(model, output_path, data_ui=(self.ref_window is None))

    def initUI(self):
        from utils.MyGLWindow import MyGLWindow

        super(OpenGLRandomEvaluationUI, self).initUI()

        if self.ref_window is None:
            opengl_widget = MyGLWindow(3)
            opengl_widget.setFixedSize(564, 150)
            self.data_lable_layout.addWidget(opengl_widget, 1, 0, 1, 3)

            self.opengl_widget = opengl_widget
        else:
            self.main_height = 100
            self.setFixedHeight(self.main_height)

    def paint(self):
        import mcubes

        super(OpenGLRandomEvaluationUI, self).paint()

        if self.opengl_widget.shaderProgram is not None:
            thres = 0.5
            if self.current_updated_flag:
                vertices, triangles = mcubes.marching_cubes(self.current_data, thres)
                self.opengl_widget.setObj(0, vertices, triangles)
                self.current_updated_flag = False
            if self.target_updated_flag:
                vertices, triangles = mcubes.marching_cubes(self.target_data, thres)
                self.opengl_widget.setObj(1, vertices, triangles, focus=True)
                self.target_updated_flag = False
            if self.current_best_updated_flag:
                vertices, triangles = mcubes.marching_cubes(self.current_best_data, thres)
                self.opengl_widget.setObj(2, vertices, triangles)
                self.current_best_updated_flag = False

            self.opengl_widget.repaint()

    def save_search_result(self):
        import mcubes

        super(OpenGLRandomEvaluationUI, self).save_search_result()

        for i in range(len(self.best_choices)):
            model_name = self.current_output_path + '/' + str(i) + '.obj'
            model_float = self.best_choices[i]['data']
            thres = 0.5
            vertices, triangles = mcubes.marching_cubes(model_float, thres)
            mcubes.export_obj(vertices, triangles, model_name)

        model_name = self.current_output_path + '/target.obj'
        model_float = self.target_data
        thres = 0.5
        vertices, triangles = mcubes.marching_cubes(model_float, thres)
        mcubes.export_obj(vertices, triangles, model_name)

    def myShow(self):
        super(OpenGLRandomEvaluationUI, self).myShow()
        if self.ref_window is not None:
            self.ref_window.show()

    def myClose(self):
        super(OpenGLRandomEvaluationUI, self).myClose()
        if self.ref_window is not None:
            self.ref_window.close()