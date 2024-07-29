#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import librosa
import matplotlib.pyplot as plt
import numpy as np
from dtw import dtw

# Loading audio files
y1, sr1 = librosa.load('/Users/rishanahata/Downloads/mfcc/data/CP-2.wav')
y2, sr2 = librosa.load('/Users/rishanahata/Downloads/mfcc/data/tts-2.wav')

# Framing parameters
frame_size = 0.025  # 25ms frame size
frame_stride = 0.05  # 50ms frame stride

# Framing the audio signals
frames1 = librosa.util.frame(y1, frame_length=int(frame_size * sr1), hop_length=int(frame_stride * sr1))
frames2 = librosa.util.frame(y2, frame_length=int(frame_size * sr2), hop_length=int(frame_stride * sr2))

# Computing MFCC for each frame
mfcc1 = np.array([librosa.feature.mfcc(y=frame, sr=sr1, n_fft=512) for frame in frames1.T])
mfcc2 = np.array([librosa.feature.mfcc(y=frame, sr=sr2, n_fft=512) for frame in frames2.T])

# Computing similarity between frames
similarity = np.zeros((len(frames1.T), len(frames2.T)))
for i, frame1 in enumerate(mfcc1):
    for j, frame2 in enumerate(mfcc2):
        similarity[i, j] = np.linalg.norm(frame1 - frame2)

# DTW alignment
dist = lambda x, y: np.linalg.norm(x - y)
dist, cost, acc_cost, path = dtw(mfcc1.mean(axis=0).T, mfcc2.mean(axis=0).T, dist=dist)

# Plotting similarity matrix and DTW alignment
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(similarity, cmap='coolwarm', interpolation='nearest')
plt.title("Similarity between Frames")
plt.xlabel("Frames of Sample 2")
plt.ylabel("Frames of Sample 1")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(acc_cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
plt.title("DTW Alignment")
plt.xlabel("Frames of Sample 2")
plt.ylabel("Frames of Sample 1")
plt.plot([j for j, _ in path], [i for i, _ in path], 'w-')
plt.xlim(-0.5, similarity.shape[1] - 0.5)
plt.ylim(-0.5, similarity.shape[0] - 0.5)

plt.tight_layout()
plt.show

