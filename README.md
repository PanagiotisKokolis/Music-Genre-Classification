# GenreClassifier

Genre Classifier is a multi-use package that allows you to:  
1- create your own music dataset to be trained by a neural network  
2- convert any copyright free mp3 to .wav file to either add to your dataset or evaluate with a model  
3- train or use an existing model to attempt to guess the genre of your music  



# Setup  
Install libraries via requirements.txt  

Genre Classifier uses the Marsyas dataset to train it's model, you can download the file (1.2GB) here:    
https://huggingface.co/datasets/marsyas/gtzan   
*Note Jazz.00054.wav is corrupted in the dataset and should be removed  

## Converting an mp3 to wav  
Run the convert_mp3_to_wav.py file and select your .mp3 file  
The file will be in the /test_output folder listed as converted_output.wav  
You can use the file to be evaluated by the model, or incorportate it in test data under genre folder  

## Preparing your dataset  
Input the path of your dataset in line 6 of the prepare_dataset.py file
Take the Marsyas (or personal) dataset and run the prepare_dataset.py file  
The dataset will be converted to a jsonfile of mfcc values usable by the neural network model  

## Create your own training model  
Using a json file of mfcc values obtained from the prepare_dataset.py file, run the create_nn_model.py file to create your own model  
    Please note this model replaces the current model (.ckpt) file in the package  
    Use:  tensorboard --logdir tb_logs/   
    to evaluate the accuracy of your model  

## Predict a Genre
Run the predict_song_genre.py file to predict the genre of a song of your choice  
You will be prompted to submit a .wav file  


TODO: environment with relevant packages --done  
TODO: implement GTZAN dataset and create genre folders with associated mfccs --done  
TODO: 80-20 train test split --done  
TODO: pytorch neural network --done   
TODO: minimize loss/error  --current accuracy rate 70% suggestions include:  
    Further data augmentation: convolutions, segmentation, increase files/genre  



