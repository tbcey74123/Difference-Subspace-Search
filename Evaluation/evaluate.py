import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import json
import numpy as np
import argparse
from PyQt5 import QtWidgets, QtCore

from utils.utils import readSearchCases

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('settings', help='settings', choices=['pggan_celebahq.json', 'gansynth.json'])
    parser.add_argument('subject', help='subject index', type=int)

    args = parser.parse_args()

    subject_index = args.subject

    fr = open(args.settings)
    evaluation_info = json.load(fr)

    if evaluation_info['dataset'] == 'PGGAN_celebahq':
        from models.PGGANWrapper import PGGANWrapper
        from EvaluationUI import ImageEvaluationUI
        from BOEvaluationUI import ImageBOEvaluationUI
        from RandomEvaluationUI import ImageRandomEvaluationUI

        weights_path = '../pretrained_weights/' + evaluation_info['weights_path']
        model = PGGANWrapper(weights_path)
    elif evaluation_info['dataset'] == 'GANSynth':
        from models.GANSynthWrapper import GANSynthWrapper
        from EvaluationUI import AudioEvaluationUI
        from BOEvaluationUI import AudioBOEvaluationUI
        from RandomEvaluationUI import AudioRandomEvaluationUI

        model = GANSynthWrapper('../pretrained_weights/' + evaluation_info['weights_path'], evaluation_info["sample_rate"])

    init_latents, target_latents, _ = readSearchCases(evaluation_info['cases_path'])
    # manually change the initial data for GANSynth
    # random_init_latents, random_target_latents, _ = readSearchCases(evaluation_info['pool_path'])
    # init_latents[0] = random_init_latents[0]
    # init_latents[1] = random_target_latents[1]
    # init_latents[2] = random_init_latents[1]
    # init_latents[3] = random_init_latents[1]
    # init_latents[4] = random_target_latents[1]
    # init_latents[5] = random_init_latents[1]
    search_cases = np.random.choice(target_latents.shape[0], target_latents.shape[0], replace=False)

    # To prevent the results got overwirtten
    # if os.path.isdir(evaluation_info['output_path'] + "time_quality_evaluation_1d/" + str(subject_index)) or \
    #     os.path.isdir(evaluation_info['output_path'] + "time_quality_evaluation_bo/" + str(subject_index)) or \
    #     os.path.isdir(evaluation_info['output_path'] + "time_quality_evaluation_random/" + str(subject_index)):
    #     exit(0)

    app = QtWidgets.QApplication(sys.argv)
    ui_windows = []
    if evaluation_info['dataset'] == 'PGGAN_celebahq':
        ui_windows.append(ImageEvaluationUI(model, evaluation_info['output_path'] + "time_quality_evaluation_1d/" + str(subject_index)))
        ui_windows.append(ImageBOEvaluationUI(model, evaluation_info['output_path'] + "time_quality_evaluation_bo/" + str(subject_index), evaluation_info['range']))
        ui_windows.append(ImageRandomEvaluationUI(model, evaluation_info['output_path'] + "time_quality_evaluation_random/" + str(subject_index)))
    elif evaluation_info['dataset'] == 'GANSynth':
        ui_windows.append(AudioEvaluationUI(model, evaluation_info['output_path'] + "time_quality_evaluation_1d/" + str(subject_index), evaluation_info['sample_rate']))
        ui_windows.append(AudioBOEvaluationUI(model, evaluation_info['output_path'] + "time_quality_evaluation_bo/" + str(subject_index), evaluation_info['sample_rate'], evaluation_info['range']))
        ui_windows.append(AudioRandomEvaluationUI(model, evaluation_info['output_path'] + "time_quality_evaluation_random/" + str(subject_index), evaluation_info['sample_rate']))
    for i in range(len(ui_windows)):
        ui_windows[i].setWindowTitle("Evaluation - " + evaluation_info['dataset'])

    first_flag = False
    for i in range(search_cases.shape[0]):
        search_case = search_cases[i]
        # Strategy 0: ours
        # Strategy 1: SLS
        # Strategy 2: randomstrategies
        strategies = np.random.choice(3, 3, replace=False)
        if first_flag is False:
            first_flag = True
            for j in range(1, len(ui_windows)):
                ui_windows[strategies[j]].close()
        for strategy in strategies:
            ui_windows[strategy].myShow()
            ui_windows[strategy].start_evaluation(init_latents[search_case], target_latents[search_case], evaluation_info['time'], search_case)
            app.exec_()
    sys.exit(0)