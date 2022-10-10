import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def librosa_pyin(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

import natsupitch
def natsu_pyin(y, sr):
    f0, voiced_flag, voiced_probs = natsupitch.core.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), device="cpu")

def natsu_pyin_cuda(y, sr):
    f0, voiced_flag, voiced_probs = natsupitch.core.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), device="cuda")

import crepe
def test_crepe(y, sr):
    time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True)

import harmof0
import torch
import torchaudio
pit = harmof0.PitchTracker(device="cpu")
pit_cuda = harmof0.PitchTracker(device="cuda")

def harmo(waveform, sr):
    time, freq, activation, activation_map = pit.pred(waveform, sr)

def harmo_cuda(waveform, sr):
    time, freq, activation, activation_map = pit_cuda.pred(waveform, sr)

TEST_NUM=3
# waveform, sr = torchaudio.load(librosa.ex('trumpet'))
waveform, sr = torchaudio.load("lty_pure.wav")

resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
waveform = resampler(waveform)

waveform = torch.mean(waveform, dim=0)
waveform_np = waveform.numpy()
sr = 16000

print("librosa PYIN")
start = time.time()
for i in range(TEST_NUM):
    librosa_pyin(waveform_np, sr)
end = time.time()
print(f"{end - start}s")

print("Optimizated PYIN (CPU)")
start = time.time()
for i in range(TEST_NUM):
    natsu_pyin(waveform_np, sr)
end = time.time()
print(f"{end - start}s")

print("Optimizated PYIN (torch CUDA)")
start = time.time()
for i in range(TEST_NUM):
    natsu_pyin_cuda(waveform_np, sr)
end = time.time()
print(f"{end - start}s")


'''
print("crepe")
start = time.time()
for i in range(TEST_NUM):
    test_crepe(waveform_np, sr)
end = time.time()
print(f"{end - start}s")
'''

print("HarmoF0")
start = time.time()
for i in range(TEST_NUM):
    harmo(waveform, sr)
end = time.time()
print(f"{end - start}s")

print("HarmoF0 (CUDA)")
start = time.time()
for i in range(TEST_NUM):
    harmo_cuda(waveform, sr)
end = time.time()
print(f"{end - start}s")