import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import natsupitch

def librosa_pyin(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return f0, voiced_flag, voiced_probs

def natsu_pyin(y, sr):
    f0, voiced_flag, voiced_probs = natsupitch.core.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), device="cuda")
    return f0, voiced_flag, voiced_probs

# y, sr = librosa.load(librosa.ex('trumpet'))
y, sr = librosa.load("lty_pure.wav")
ret0 = librosa_pyin(y, sr)
ret1 = natsu_pyin(y, sr)
ret = np.allclose(ret0[0], ret1[0], equal_nan=True)
print(ret)
print([i for i in zip(ret0[0], ret1[0]) if abs(i[0] - i[1]) > 0])