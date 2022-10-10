import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import natsupitch

def show_example():
    y, sr = librosa.load(librosa.ex('trumpet'))
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')
    plt.show()

# show_example()

def librosa_pyin(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

def natsu_pyin(y, sr):
    f0, voiced_flag, voiced_probs = natsupitch.core.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    '''
    times = librosa.times_like(f0)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')
    plt.show()
    '''

import harmof0
import torchaudio
pit = harmof0.PitchTracker()

def harmo(waveform, sr):
    time, freq, activation, activation_map = pit.pred(waveform, sr)

waveform, sr = torchaudio.load(librosa.ex('trumpet'))

resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
waveform = resampler(waveform)
waveform_np = waveform.numpy()
sr = 16000

print("librosa PYIN")
start = time.time()
for i in range(10):
    librosa_pyin(waveform_np, sr)
end = time.time()
print(end - start)

print("Optimizated PYIN")
start = time.time()
for i in range(10):
    natsu_pyin(waveform_np, sr)
end = time.time()
print(end - start)

print("HarmoF1")
start = time.time()
for i in range(10):
    harmo(waveform, sr)
end = time.time()
print(end - start)