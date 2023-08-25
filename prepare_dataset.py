import json
import os
import math
import librosa

DATASET_PATH = "/Users/pana/Downloads/genres"
JSON_PATH = "mfcc_data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=3):
    """Creates MFCC and labels it according to genre. Defaults listed.

        :param dataset_path (str): Marsyas dataset path
        :param json_path (str): Path of created fileset
        :param num_mfcc (int): mfccs per segment
        :param n_fft (int): FFT interval
        :param hop_length (int): sliding window for FFT
        :param: num_segments (int): split tracks into smaller segments to increase model training set
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # catch case between folders
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for wav in filenames:

		        # load wav
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
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, j+1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
               
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=3)