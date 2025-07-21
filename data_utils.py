"""PPG Data loading and preparation for ML model training"""

import zipfile
import numpy as np
import pandas as pd
import os


def unzip_data(path_2_zip_folder):
    f = path_2_zip_folder.replace(".zip", '')
    with zipfile.ZipFile(path_2_zip_folder, 'r') as zip_ref:
        zip_ref.extractall(f)


def check_if_files_unzipped(data_path):
    if not os.path.exists(data_path):
        all_zipped_files = [x for x in os.listdir(os.getcwd()) if '.zip' in x]
        for f in all_zipped_files:
            fpath = "{0}/{1}".format(os.getcwd(), f)
            unzip_data(fpath)
    else:
        print("Data already unzipped")
        print("Data path exists:", data_path)


def create_dataset(X, y, step, time_steps):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs, np.float32), np.array(ys)


# cogWear pilot data processing

def cogwear_pilot_load_participant(ppath, i, device='empatica', physioparam="bvp", sampling_frequency=64):
    sec2 = sampling_frequency * 2  # remove 2 seconds from start and end of the experimental condition
    clT = pd.read_csv('{0}/{1}/cognitive_load/{2}_{3}.csv'.format(ppath, i, device, physioparam))
    baseT = pd.read_csv('{0}/{1}/baseline/{2}_{3}.csv'.format(ppath, i, device, physioparam))
    return clT[physioparam].values[sec2:-sec2], baseT[physioparam].values[sec2:-sec2]


def cogwear_pilot_prepare_participant_data(cl, base, step, window, train):
    # step for baseline is adjusted to create balance number of samples from both classes
    stepB = int(step / 2) if train else step
    xc, yc = create_dataset(cl, np.ones(len(cl)), step=step, time_steps=window)
    xb, yb = create_dataset(base, np.zeros(len(base)), step=stepB, time_steps=window)
    x = np.concatenate([xc, xb])
    y = np.concatenate([yc, yb])
    return x, y


def cogwear_pilot_load_dataset(ppath, indicies, step, window, device, physioparam, sampling, train=True):
    X, Y = [], []
    for i in indicies:
        cl, base = cogwear_pilot_load_participant(ppath, i, device, physioparam, sampling)
        x, y = cogwear_pilot_prepare_participant_data(cl, base, step, window, train)
        X.append(x)
        Y.append(y)
    return np.concatenate(X), np.concatenate(Y)


def cogwear_survey_load_participant_calibration(ppath, i, cond, device='empatica', sampling_frequency=64):
    sec2 = sampling_frequency * 2  # remove 2 seconds from start and end of the experimental condition
    clT = pd.read_csv('{0}/{1}/{2}/cognitive_load/{3}_bvp.csv'.format(ppath, i, cond, device))
    baseT = pd.read_csv('{0}/{1}/{2}/baseline/{3}_bvp.csv'.format(ppath, i, cond, device))
    return clT.bvp.values[sec2:-sec2], baseT.bvp.values[sec2:-sec2]



