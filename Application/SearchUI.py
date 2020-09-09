import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from PyQt5 import QtWidgets, QtCore, QtMultimedia
from utils.MySlider import MySlider
from utils.utils import *
from utils.QtPlot import QtPlot
from utils.MyDialog import MyDialog

from enum import Enum
class State(Enum):
    UNINITIALIZED = 0
    INITIALIZING = 1
    SEARCHING = 2

class SearchUI(QtWidgets.QWidget):
    def __init__(self, model, output_path):
        super(SearchUI, self).__init__()
        self.model = model
        self.output_base_path = output_path

        # Dimension Parameters
        self.latent_size = model.latent_size
        self.data_size = model.data_size
        self.data_dim = model.data_dim

        # Jacobian
        self.jacobian_s_thres = 1e-20

        self.jacobian_vhs = None
        self.jacobian_s = None
        self.jacobian_mask = None

        # Subspace
        # self.subspace_range = 2.25
        self.subspace_range = self.model.expected_distance * 0.2
        # self.subspace_range = 7.5

        self.subspace_basis = None

        # User Interactions Variables
        self.center_tolerance = 0.05
        self.predict_user = None
        self.predict_system = None
        self.current_data = None
        self.target_data = None
        self.center_data = None
        self.center_latent = None
        self.current_best_data = None
        self.search_current_n = None
        self.search_output_base = None
        self.current_subspace_idx = 0
        self.cached_search_t = None

        self.current_updated_flag = False

        self.current_state = State.UNINITIALIZED

        # UI
        self.initUI()

        self.show()

    def initUI(self):
        # self.main_width = 250
        # self.main_height = 250
        self.main_width = 380
        self.main_height = 325
        self.setGeometry(600, 515, self.main_width, self.main_height)

        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)

        self.top_spacer = QtWidgets.QSpacerItem(0, 0)
        main_layout.addSpacerItem(self.top_spacer)

        self.data_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(self.data_layout)

        self.bottom_spacer = QtWidgets.QSpacerItem(0, 0)
        main_layout.addSpacerItem(self.bottom_spacer)

        self.search_slider = MySlider()
        self.search_slider.setMinimum(0)
        self.search_slider.setMaximum(1000)
        self.search_slider.setValue(500)
        self.search_slider.setTickInterval(1)
        self.search_slider.sliderReleased.connect(self.OnSearchSliderReleased)
        main_layout.addWidget(self.search_slider)

        tmp_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(tmp_layout)
        self.button = QtWidgets.QPushButton()
        self.button.setStyleSheet("QPushButton{border-image:url(resources/next.png);} QPushButton:pressed{border-image:url(resources/next_p.png);}")
        self.button.setFixedSize(85, 45)
        self.button.clicked.connect(self.OnButtonPressed)
        tmp_layout.addWidget(self.button)

        # self.save_button = QtWidgets.QPushButton()
        # self.save_button.setStyleSheet("QPushButton{border-image:url(resources/save.png);} QPushButton:pressed{border-image:url(resources/save_p.png);}")
        # self.save_button.setFixedSize(42, 42)
        # self.save_button.clicked.connect(self.OnSaveButtonPressed)
        # tmp_layout.addWidget(self.save_button)

    def start_search(self, init_latent = None):
        if init_latent is not None:
            self.switch_state(State.SEARCHING)

            self.center_latent = init_latent
            self.update_subspace(self.center_latent)
            self.search_slider.setValue(500)
            self.sample_new_search_point()
        else:
            self.switch_state(State.INITIALIZING)
            self.sample_new_initial_point()
        self.paint()

    def switch_state(self, new_state):
        self.current_state = new_state
        if self.current_state == State.INITIALIZING:
            self.search_slider.interactable = False
            self.button.setStyleSheet("QPushButton{border-image:url(resources/start.png);} QPushButton:pressed{border-image:url(resources/start_p.png);}")
        elif self.current_state == State.SEARCHING:
            self.search_slider.interactable = True
            self.button.setStyleSheet("QPushButton{border-image:url(resources/next.png);} QPushButton:pressed{border-image:url(resources/next_p.png);}")

    def sample_new_initial_point(self):
        self.current_latent = self.model.get_random_latent()
        self.current_data = self.model.decode(self.current_latent.reshape(1, -1))[0]
        self.current_updated_flag = True

    def paint(self):
        pass

    def update_subspace(self, p):
        self.jacobian = self.model.calc_model_gradient(p.reshape(1, self.latent_size))
        u, s, vh = np.linalg.svd(self.jacobian, full_matrices=True)

        num = s.shape[0]

        self.jacobian_vhs = vh[:num]
        self.jacobian_s = s[:num]
        self.jacobian_mask = np.ones(self.jacobian_vhs.shape[0], dtype=bool)

        self.importance_sampling_subspace_basis()
        print('end updating subspace!')

    def importance_sampling_subspace_basis(self):
        if self.jacobian_s[self.jacobian_mask].shape[0] <= 0:
            self.jacobian_mask = np.ones(self.jacobian_vhs.shape[0], dtype=bool)

        s = self.jacobian_s[self.jacobian_mask] + 1e-6
        vh = self.jacobian_vhs[self.jacobian_mask]

        # Importance sampling
        choice_p = s / np.sum(s)
        idx = np.random.choice(choice_p.shape[0], p=choice_p)

        self.subspace_basis = vh[idx]

        self.jacobian_mask[np.arange(self.jacobian_vhs.shape[0])[self.jacobian_mask][idx]] = False

        print('Finish sampling subspace!!')

    def get_slider_point(self, t):
        new_point = (self.center_latent - self.subspace_basis * self.subspace_range) * (1 - t) + (self.center_latent + self.subspace_basis * self.subspace_range) * t
        return new_point

    def sample_new_search_point(self):
        t = self.search_slider.value() / 1000.0
        self.current_latent = self.get_slider_point(t)
        self.current_data = self.model.decode(self.current_latent.reshape(1, -1))[0]
        self.current_updated_flag = True

    def save_search_result(self):
        safe_mkdir(self.output_base_path)

        self.output_n = len(os.listdir(self.output_base_path))

    def OnSearchSliderReleased(self):
        self.sample_new_search_point()
        self.paint()

    def OnButtonPressed(self):
        dialog = MyDialog(self)
        dialog.show()
        QtCore.QCoreApplication.processEvents()
        if self.current_state == State.INITIALIZING:
            self.start_search(self.current_latent)
        elif self.current_state == State.SEARCHING:
            if np.abs(self.search_slider.value() / 1000.0 - 0.5) < self.center_tolerance:
                self.importance_sampling_subspace_basis()
            else:
                self.center_latent = self.current_latent
                self.update_subspace(self.center_latent)

            self.search_slider.setValue(500)
            self.sample_new_search_point()
        dialog.destroy()

    def OnSaveButtonPressed(self):
        self.save_search_result()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_S:
            self.OnSaveButtonPressed()
        elif event.key() == QtCore.Qt.Key_R:
            self.switch_state(State.INITIALIZING)
            self.sample_new_initial_point()

            self.paint()

class AudioSearchUI(SearchUI):
    def __init__(self, model, output_path, sr):
        super(AudioSearchUI, self).__init__(model, output_path)

        # Audio Player Parameters
        self.volume_param = 20000
        self.sr = sr

        # Audio Player
        self.initAudioPlayer()

    def initUI(self):
        super(AudioSearchUI, self).initUI()

        self.top_spacer.changeSize(0, 25)
        self.bottom_spacer.changeSize(0, 25)

        self.play_sound_button = QtWidgets.QPushButton()
        self.play_sound_button.setStyleSheet("QPushButton{border-image:url(resources/play_sound_square.png);}")
        self.play_sound_button.setFixedSize(150, 150)
        self.play_sound_button.clicked.connect(self.OnPlaySoundButtonClicked)
        self.data_layout.addWidget(self.play_sound_button)

        # self.top_spacer.changeSize(0, 62.5)
        # self.bottom_spacer.changeSize(0, 62.5)
        #
        # height = 150
        # width = self.main_width - 50
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        # # sizePolicy.setHeightForWidth(True)
        # self.current_wave_canvas = QtPlot()
        # self.current_wave_canvas.setSizePolicy(sizePolicy)
        # self.current_wave_canvas.setFixedSize(width, height)
        # self.data_layout.addWidget(self.current_wave_canvas)

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
        super(AudioSearchUI, self).save_search_result()

        sound = self.current_data
        sound_name = str(self.output_n) + '.wav'
        self.write_wave(sound, self.output_base_path + '/' + sound_name, self.sr)

    def write_wave(self, wave, name, sr):
        import librosa

        librosa.output.write_wav(name, wave.astype(np.float32), sr)

    def OnPlaySoundButtonClicked(self):
        if self.current_data is None:
            return
        sound = self.current_data
        self.play_sound(sound)

    def OnSearchSliderReleased(self):
        super(AudioSearchUI, self).OnSearchSliderReleased()
        sound = self.current_data
        self.play_sound(sound)

    # def paint(self):
    #     super(AudioSearchUI, self).paint()
    #
    #     self.current_wave_canvas.clear()
    #
    #     if self.current_data is not None:
    #         self.current_wave_canvas.plot(np.arange(self.current_data.shape[0]), self.current_data)
    #         self.current_wave_canvas.draw()

class ImageSearchUI(SearchUI):
    def __init__(self, model, output_path):
        super(ImageSearchUI, self).__init__(model, output_path)

    def initUI(self):
        super(ImageSearchUI, self).initUI()

        size = self.main_height - 50
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        # sizePolicy.setHeightForWidth(True)
        self.current_image_canvas = QtPlot()
        self.current_image_canvas.setSizePolicy(sizePolicy)
        self.current_image_canvas.setFixedSize(size, size)
        self.data_layout.addWidget(self.current_image_canvas)

    def paint(self):
        super(ImageSearchUI, self).paint()

        self.current_image_canvas.clear()

        cmap = 'gray'
        if self.data_dim != 2:
            cmap = None

        if self.current_data is not None:
            self.current_image_canvas.imshow(self.current_data, cmap)
            self.current_image_canvas.draw()

    def save_search_result(self):
        import cv2

        super(ImageSearchUI, self).save_search_result()

        img = self.current_data
        if self.data_dim != 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fig_name = str(self.output_n) + '.png'
        cv2.imwrite(self.output_base_path + '/' + fig_name, img)

class OpenGLSearchUI(SearchUI):
    def __init__(self, model, output_path):
        super(OpenGLSearchUI, self).__init__(model, output_path)

    def initUI(self):
        from utils.MyGLWindow import MyGLWindow

        super(OpenGLSearchUI, self).initUI()

        size = self.main_height - 50
        self.opengl_widget = MyGLWindow(1)
        self.opengl_widget.setFixedSize(size, size)
        self.data_layout.addWidget(self.opengl_widget)

    def paint(self):
        import mcubes

        super(OpenGLSearchUI, self).paint()

        if self.opengl_widget.shaderProgram is not None:
            thres = 0.5
            if self.current_updated_flag:
                vertices, triangles = mcubes.marching_cubes(self.current_data, thres)
                self.opengl_widget.setObj(0, vertices, triangles, focus=True)
                self.current_updated_flag = False

            self.opengl_widget.repaint()

    def save_search_result(self):
        import mcubes

        super(OpenGLSearchUI, self).save_search_result()

        model_name = str(self.output_n) + '.obj'
        model_float = self.current_data
        thres = 0.5
        vertices, triangles = mcubes.marching_cubes(model_float, thres)
        mcubes.export_obj(vertices, triangles, self.output_base_path + '/' + model_name)
