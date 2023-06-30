import glob
import os

import numpy as np
import pandas as pd

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
        os.path.join(dirname, "all_features.txt"), dtype=str, comments="#"
    )

    data_x = data[:, 2:].astype("float")
    data_y = data[:, 0]

    model = ensemble_model.Classifier(data_x, data_y, n_iter=25, boruta_trials=25, impute=True)
    model.create()
    model.save()


def predict_ogle(model, path="OGLE_II", suffix="*.dat"):
    filenames = get_data_filenames(path, suffix)
    this_file = filenames[0]
    time, mag, magerr = np.loadtxt(this_file, unpack=True)

    prediction = model.predict(time, mag, magerr, convert=True, zp=22)

    return prediction


def run(csv_file=os.path.join(os.getenv("HOME"), "MicroLIA_Training_Set.csv")):
#    train()
    df = pd.read_csv(csv_file)
    # The keyword name is "csv_file", but it wants a Pandas DataFrame
    model = ensemble_model.Classifier(csv_file=df)
    model.load()
    prediction = predict_ogle(model)
    print(prediction)


if __name__ == "__main__":
    run()
