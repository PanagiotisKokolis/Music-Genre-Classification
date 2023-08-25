import os
import math
import librosa
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "ENTER DATASET PATH HERE" # Change this to the location of the audio folders.
TRAINING_OUTPUT_PATH = "gtzan_data_train.h5"
VALIDATION_OUTPUT_PATH = "gtzan_data_val.h5"
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, training_path, validation_path, val_size=0.2, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Creates MFCCs and labels it according to genre. Saves to training, validation, and testing data. Defaults listed.

        :param dataset_path (str): Dataset path for audio files.
        :param training_path (str): Path for the training data output.
        :param validation_path (str): Path for the validation data output.
        :param val_size (float): Ratio of the data that should be set to validation data.
        :param num_mfcc (int): MFCCs per segment.
        :param n_fft (int): FFT interval.
        :param hop_length (int): Sliding window for FFT.
        :param: num_segments (int): Split tracks into smaller segments to increase model training set.
        :return:
    """
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = SAMPLES_PER_TRACK // num_segments
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # catch case between folders
        if dirpath is not dataset_path:
            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for wav in filenames:
                file_path = os.path.join(dirpath, wav)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for j in range(num_segments):
                    # calculate start and finish sample for current segment
                    start = samples_per_segment * j
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc)
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, j+1))

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(np.array(data["mfcc"]), np.array(data["labels"]), test_size=val_size)

    # Save to HDF5 for training data
    with h5py.File(training_path, 'w') as h5f:
        h5f.create_dataset('mfcc', data=X_train)
        h5f.create_dataset('labels', data=y_train)
        h5f.attrs["mapping"] = np.array(data["mapping"], dtype=h5py.special_dtype(vlen=str))

    # Save to HDF5 for validation data
    with h5py.File(validation_path, 'w') as h5f:
        h5f.create_dataset('mfcc', data=X_val)
        h5f.create_dataset('labels', data=y_val)
        h5f.attrs["mapping"] = np.array(data["mapping"], dtype=h5py.special_dtype(vlen=str))

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, TRAINING_OUTPUT_PATH, VALIDATION_OUTPUT_PATH)
