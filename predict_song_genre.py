import torch
from wav_to_mfcc import wav_to_mfcc
from create_nn_model import MusicModel
from collections import Counter

# Hard coded genres based on the training data
genre_mapping = {
    0: "Blues",
    1: "Classical",
    2: "Country",
    3: "Disco",
    4: "Hiphop",
    5: "Jazz",
    6: "Metal",
    7: "Pop",
    8: "Reggae",
    9: "Rock"
}

def predict_genre(model_path):
    """
    Predict the genre of the given .wav file using a pre-trained model.

    Args:
    - wav_file (str): Path to the .wav file.
    - model_path (str): Path to the saved model checkpoint.

    Returns:
    - str: Predicted genre.
    """
    
    mfccs = wav_to_mfcc() 
    if mfccs is None:
        print("No file was selected. Exiting")
        exit(0)
    
    model = MusicModel.load_from_checkpoint(checkpoint_path=model_path)
    model.eval()

    predicted_genres = []
    with torch.no_grad():
        for mfcc in mfccs:
            input_tensor = torch.tensor(mfcc).float().reshape(1, -1)
            predictions = model(input_tensor)
            predicted_class = torch.argmax(predictions).item()
            predicted_genres.append(genre_mapping[predicted_class])

    most_common_genre = Counter(predicted_genres).most_common(1)[0][0]
    return most_common_genre
    

if __name__ == "__main__":
    model_path = "model_checkpoint.ckpt"
    predicted_genre = predict_genre(model_path)
    print(f"The predicted genre is: {predicted_genre}")
