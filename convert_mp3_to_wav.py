import pydub


def convert_mp3_to_wav(src, dst):
    sound = pydub.AudioSegment.from_mp3(src)
    sound.export(dst, format='wav')

if __name__ == '__main__':
    src = 'test_input/test_30_seconds.mp3'
    dst = 'test_output/test_30_seconds_output.wav'
    convert_mp3_to_wav(src, dst)