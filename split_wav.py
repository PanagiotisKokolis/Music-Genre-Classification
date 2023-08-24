from pydub import AudioSegment
from convert_mp3_to_wav import convert_mp3_to_wav
import os

def split_wav(input_wav, output_dir, sample_duration = 30000):
    audio = AudioSegment.from_wav(input_wav)
    total_duration = len(audio)
    num_samples = total_duration // sample_duration
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        start_time = i * sample_duration
        end_time = (i + 1) * sample_duration
        sample = audio[start_time:end_time]
        
        output_path = os.path.join(output_dir, f'sample{i+1}.wav')
        sample.export(output_path, format='wav')


if __name__ == '__main__':
    input_mp3 = 'test_input/test_full_song.mp3'
    converted_wav = 'test_output/test_full_song.wav'
    
    convert_mp3_to_wav(input_mp3, converted_wav)
    split_wav(converted_wav, 'full_song_split_output')