import os
import math
import h5py
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

GTZAN_DATASET_PATH = "Enter dataset path here"
FMA_DATASET_PATH = "Enter dataset path here"
GTZAN_OUTPUT_PATH = "Enter path you want you training and validation data to save to"
FMA_OUTPUT_PATH = "Enter path you want you training and validation data to save to"
TRACKS_CSV_PATH = "Enter the .csv here (only req for fma data set)"
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def process_audio_file(file_path, num_segments, num_mfcc, n_fft, hop_length):
    signal = load_audio_file(file_path)
    if signal is None:
        return []
    return extract_mfcc_from_audio(signal, num_mfcc, n_fft, hop_length, num_segments)


def load_audio_file(file_path):
    try:
        signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        print(f"Loaded track: {file_path}")
        return signal
    except Exception as e:
        print(f"Couldn't process track {file_path}. Error: {e}")
        return None


def extract_mfcc_from_audio(signal, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    samples_per_segment = SAMPLES_PER_TRACK // num_segments
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    mfcc_data = []

    for j in range(num_segments):
        start = samples_per_segment * j
        finish = start + samples_per_segment
        segment = signal[start:finish]
        
        # Adjust n_fft if segment is too short
        if len(segment) < n_fft:
            n_fft = len(segment)
        
        mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        if len(mfcc) == num_mfcc_vectors_per_segment:
            mfcc_data.append(mfcc.tolist())
    return mfcc_data


def pitch_shift(signal, sr):
    n_steps = np.random.randint(-2, 3)
    augmented_signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
    return np.interp(np.linspace(0, len(signal)-1, len(signal)), np.arange(len(augmented_signal)), augmented_signal)

def add_noise(signal, sr):
    noise = np.random.randn(*signal.shape) * 0.005
    return signal + noise

def speed_up(signal):
    speed_factor = np.random.uniform(0.9, 1.1)
    augmented_signal = librosa.effects.time_stretch(signal, rate=speed_factor)
    return np.interp(np.linspace(0, len(signal)-1, len(signal)), np.arange(len(augmented_signal)), augmented_signal)



def augment_data(signal, sr):
    functions = [pitch_shift, add_noise, speed_up]
    func = np.random.choice(functions)
    if func == speed_up:
        augmented_signal = func(signal)
    else:
        augmented_signal = func(signal, sr)
    return augmented_signal



def save_data_split(hdf5_base_path, mappings, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    #Save the data split into training, validation, and optionally test HDF5 files
    save_data_to_hdf5(f"{hdf5_base_path}_train.h5", X_train, y_train, mappings)
    save_data_to_hdf5(f"{hdf5_base_path}_val.h5", X_val, y_val, mappings)
    if X_test is not None and y_test is not None:
        save_data_to_hdf5(f"{hdf5_base_path}_test.h5", X_test, y_test, mappings)


def save_data_to_hdf5(hdf5_path, mfccs, labels, mappings):
    with h5py.File(hdf5_path, "w") as hf:
        hf.create_dataset("mfcc", data=mfccs)
        hf.create_dataset("labels", data=labels)
        hf.attrs["mapping"] = np.array(mappings, dtype=h5py.special_dtype(vlen=str))
        print(f"Data saved to: {hdf5_path}")


def process_gtzan_mfccs(dataset_path, hdf5_base_path, val_size=0.2, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "train_mfcc": [],
        "train_labels": [],
        "val_mfcc": [],
        "val_labels": []
    }

    # loop through all genre folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # catch case between folders
        if dirpath != dataset_path:
            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            genre_mfccs = []
            genre_labels = []

            # process all audio files in genre sub-dir
            for wav in filenames:
                file_path = os.path.join(dirpath, wav)
                signal = load_audio_file(file_path)
                if signal is None:
                    continue

                # Extract MFCCs from the original signal
                track_mfccs = extract_mfcc_from_audio(signal, num_mfcc, n_fft, hop_length, num_segments)
                genre_mfccs.extend(track_mfccs)
                genre_labels.extend([i-1] * len(track_mfccs))

                # Augment the raw audio signal
                augmented_signal = augment_data(signal, SAMPLE_RATE)
                augmented_mfcc = extract_mfcc_from_audio(augmented_signal, num_mfcc, n_fft, hop_length, num_segments)
                genre_mfccs.extend(augmented_mfcc)
                genre_labels.extend([i-1] * len(augmented_mfcc))

            # Split data into train and validation sets for each genre
            X_train, X_val, y_train, y_val = train_test_split(np.array(genre_mfccs), np.array(genre_labels), test_size=val_size, stratify=genre_labels)

            # Extend the main data dictionary
            data["train_mfcc"].extend(X_train)
            data["train_labels"].extend(y_train)
            data["val_mfcc"].extend(X_val)
            data["val_labels"].extend(y_val)

    # Save the training and validation data in separate files
    save_data_split(f"{hdf5_base_path}GenreClassifier", data["mapping"], np.array(data["train_mfcc"]), np.array(data["train_labels"]), np.array(data["val_mfcc"]), np.array(data["val_labels"]))


def process_fma_mfccs(dataset_path, hdf5_base_path, tracks_csv_path, subset, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    tracks = pd.read_csv(tracks_csv_path, index_col=0, header=[0, 1])
    tracks_subset = tracks[tracks[('set', 'subset')] == subset]

    unique_genres = tracks_subset['track', 'genre_top'].dropna().unique()

    data = {
        "training": {
            "mapping": list(unique_genres),
            "labels": [],
            "mfcc": []
        },
        "validation": {
            "mapping": list(unique_genres),
            "labels": [],
            "mfcc": []
        },
        "test": {
            "mapping": list(unique_genres),
            "labels": [],
            "mfcc": []
        }
    }

    splits = ["training", "validation", "test"]
    for split in splits:
        tracks_split = tracks_subset[tracks_subset[('set', 'split')] == split]
        for track_id, row in tracks_split.iterrows():
            genre = row['track', 'genre_top']
            genre_idx = data[split]["mapping"].index(genre)
            if pd.isna(genre) or genre not in data[split]["mapping"]:
                continue
            
            tid_str = '{:06d}'.format(track_id)
            file_path = os.path.join(dataset_path, tid_str[:3], tid_str + '.mp3')

            track_mfccs = process_audio_file(file_path, num_segments, num_mfcc, n_fft, hop_length)
            data[split]["mfcc"].extend(track_mfccs)
            data[split]["labels"].extend([genre_idx] * len(track_mfccs))
            
            #tempcounter = Counter(data[split]['labels'])
            #print(f"Labels count for {split}: {tempcounter}")   debug print
            #print(f"{split} samples: {len(data[split]['mfcc'])}") debug print

        # Save data split to HDF5
        save_data_to_hdf5(
            f"{hdf5_base_path}_{split}.h5", 
            np.array(data[split]["mfcc"]), 
            np.array(data[split]["labels"]), 
            data[split]["mapping"]
        )

#WIP version, integrating data augmentation      
# def process_fma_mfccs(dataset_path, hdf5_base_path, tracks_csv_path, subset, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
#     tracks = pd.read_csv(tracks_csv_path, index_col=0, header=[0, 1])
#     tracks_subset = tracks[tracks[('set', 'subset')] == subset]

#     unique_genres = tracks_subset['track', 'genre_top'].dropna().unique()

#     data = {
#         "training": {
#             "mapping": list(unique_genres),
#             "labels": [],
#             "mfcc": []
#         },
#         "validation": {
#             "mapping": list(unique_genres),
#             "labels": [],
#             "mfcc": []
#         },
#         "test": {
#             "mapping": list(unique_genres),
#             "labels": [],
#             "mfcc": []
#         }
#     }

#     splits = ["training", "validation", "test"]
#     for split in splits:
#         tracks_split = tracks_subset[tracks_subset[('set', 'split')] == split]
#         for track_id, row in tracks_split.iterrows():
#             genre = row['track', 'genre_top']
#             genre_idx = data[split]["mapping"].index(genre)
#             if pd.isna(genre) or genre not in data[split]["mapping"]:
#                 continue
            
#             tid_str = '{:06d}'.format(track_id)
#             file_path = os.path.join(dataset_path, tid_str[:3], tid_str + '.mp3')
#             signal = load_audio_file(file_path)

#             if signal is None:
#                 continue

#             track_mfccs = extract_mfcc_from_audio(signal, num_mfcc, n_fft, hop_length, num_segments)

#             # Check if the segment is empty or too short for FFT
#             # if len(track_mfccs) != num_segments-1:
#             #     print(f"Incorrect segment length, skipping FFT. Length: {len(track_mfccs)}")
#             #     continue

#             data[split]["mfcc"].extend(track_mfccs)
#             data[split]["labels"].extend([genre_idx] * len(track_mfccs))

#             # Augment data for training set
#             if split == "training":
#                 augmented_signal = augment_data(signal, SAMPLE_RATE)  # Convert to numpy array here

#                 augmented_mfcc = extract_mfcc_from_audio(augmented_signal, num_mfcc, n_fft, hop_length, num_segments)
#                 data[split]["mfcc"].extend(augmented_mfcc)
#                 data[split]["labels"].extend([genre_idx] * len(augmented_mfcc))

#         # Save data split to HDF5
#         save_data_to_hdf5(
#             f"{hdf5_base_path}_{split}.h5", 
#             np.array(data[split]["mfcc"]), 
#             np.array(data[split]["labels"]), 
#             data[split]["mapping"]
#         )




if __name__ == "__main__":
    process_gtzan_mfccs(GTZAN_DATASET_PATH, GTZAN_OUTPUT_PATH)
    #process_fma_mfccs(FMA_DATASET_PATH, FMA_OUTPUT_PATH, TRACKS_CSV_PATH, 'small')
    #function currently has 2 calls, integrating data augmentation to the dma dataset