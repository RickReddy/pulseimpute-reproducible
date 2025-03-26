import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats


fig, ax = plt.subplots()
file = "eegfmri_nih_ppg_MSE.csv"

df = pd.read_csv("eegfmri_nih_ppg_MSE.csv", header=None, index_col=False)
for index, row in df.iterrows():
    plt.plot(row, linestyle='-', marker='o', alpha=0.5)

plt.show()
