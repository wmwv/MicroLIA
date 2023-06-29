import glob
import os

import numpy as np

from MicroLIA import ensemble_model, training_set


def get_data_filenames(path="OGLE_II", suffix="*.dat"):
    data_file_regex = os.path.join(path, suffix)
    filenames = glob.glob(data_file_regex)

    return filenames


def load_ogle_timestamps(path="OGLE_II", suffix="*.dat"):
    filenames = get_data_filenames(path, suffix)

    timestamps = []
    for file in filenames:
        data = np.loadtxt(file)
        time = data[:, 0]
        timestamps.append(time)

    return timestamps


def train(dirname=os.getenv("HOME"), name="test"):
    timestamps = load_ogle_timestamps()

    data_x, data_y = training_set.create(timestamps, min_mag=15, max_mag=20, n_class=50)

    # By default the file is saved in the home directory
    data = np.loadtxt(
        os.path.join(dirname, "all_features_.txt"), dtype=str, comments="#"
    )

    data_x = data[:, 2:].astype("float")
    data_y = data[:, 0]

    model = ensemble_model.Classifier(data_x, data_y, n_iter=25, boruta_trials=25)
    model.create(filename=name)
    model.save(filename=name)


def predict_ogle(model, path="OGLE_II", suffix="*.dat"):
    filenames = get_data_filenames(path, suffix)
    file = filenames[0]
    time, mag, magerr = np.loadtxt(file, unpack=True)

    prediction = model.predict(time, mag, magerr, convert=True, zp=22)

    return prediction


def run(csv_file=os.path.join(os.getenv("HOME"), "MicroLIA_Training_Set_test_.csv")):
#    train()
    model = ensemble_model.Classifier(csv_file)
    predict_ogle(model)


if __name__ == "__main__":
    run()
