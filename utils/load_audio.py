import numpy as np
import os
import scipy.io.wavfile as wav
from pydub import AudioSegment

def load_audio(file_name):

    if '.mp3' in file_name:
        AudioSegment.from_mp3(file_name).export(os.path.dirname(file_name), format="wav")
        rate, audio = wav.read(file_name.replace(".mp3", ".wav"))
    elif '.wav' in file_name:
        rate, audio = wav.read(file_name)
    else:
        raise ValueError('Can not open the provided file extension')
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
    return rate, audio