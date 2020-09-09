import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import argparse
import numpy as np

from PyQt5 import QtWidgets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='desired dataset', choices=['MNIST', 'PGGAN_celebahq', 'GANSynth', 'IMGAN_flight', 'IMGAN_chair'])
    parser.add_argument('--output_path', help='where the saved results would be put', default='.')
    parser.add_argument('--initial', help='intial file')
    args = parser.parse_args()

    init_latent = None
    if args.initial:
        fr = open(args.initial, 'r')
        latent_size = int(fr.readline())

        init_latent = []
        line = fr.readline()
        latent = line.split(' ')
        for j in range(latent_size):
            init_latent.append(float(latent[j]))
        init_latent = np.array(init_latent)

    dataset = args.dataset
    output_path = args.output_path
    title = ""
    if dataset == 'MNIST':
        from SearchUI import ImageSearchUI
        from models.MNISTGenerator import MNISTGenerator

        model = MNISTGenerator()
        weights_path = '../pretrained_weights/MNIST/model.ckpt-155421'
        model.load_model(weights_path)
        title = "MNIST"
    elif dataset == 'PGGAN_celebahq':
        from SearchUI import ImageSearchUI
        from models.PGGANWrapper import PGGANWrapper

        weights_path = '../pretrained_weights/PGGAN/karras2018iclr-celebahq-1024x1024.pkl'
        model = PGGANWrapper(weights_path)
        title = "PG-GAN"
    elif dataset == 'GANSynth':
        from SearchUI import AudioSearchUI
        from models.GANSynthWrapper import GANSynthWrapper

        model = GANSynthWrapper('../pretrained_weights/GANSynth/acoustic_only', 16000)
        title = "GANSynth"
    elif dataset == 'IMGAN_flight':
        from SearchUI import OpenGLSearchUI
        from models.IMGAN import IMGAN

        model = IMGAN()
        weights_path1 = '../pretrained_weights/IMGAN_flight/02691156_vox128_z_128_128/ZGAN.model-10000'
        weights_path2 = '../pretrained_weights/IMGAN_flight/02691156_vox128_64/IMAE.model-194'
        model.load_model(weights_path1, weights_path2)
        title = "IM-GAN"
    elif dataset == 'IMGAN_chair':
        from SearchUI import OpenGLSearchUI
        from models.IMGAN import IMGAN

        model = IMGAN()
        weights_path1 = '../pretrained_weights/IMGAN_chair/03001627_vox_z_128_128/ZGAN.model-10000'
        weights_path2 = '../pretrained_weights/IMGAN_chair/03001627_vox_64/IMAE.model-191'
        model.load_model(weights_path1, weights_path2)
        title = "IM-GAN"

    app = QtWidgets.QApplication(sys.argv)
    if dataset == 'MNIST' or dataset == 'PGGAN_celebahq':
        ui_window = ImageSearchUI(model, output_path)
    elif dataset == 'GANSynth':
        ui_window = AudioSearchUI(model, output_path, 16000)
    elif dataset == 'IMGAN_flight' or dataset == 'IMGAN_chair':
        ui_window = OpenGLSearchUI(model, output_path)
    # ui_window.setWindowTitle(title)
    ui_window.setWindowTitle("Differential Subspace Search")

    ui_window.start_search(init_latent)
    app.exec_()
    sys.exit(0)