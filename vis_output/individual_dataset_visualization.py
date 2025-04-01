import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats


fig, ax = plt.subplots()
MSE_file = "eegfmri_nih_ppg_MSE.csv"
timestamps_file = "eegfmri_nih_ppg_timestamps.csv"

MSE_df = pd.read_csv(MSE_file, header=None, index_col=False)
timestamps_df = pd.read_csv(timestamps_file, header=None, index_col=False)
for index, row in MSE_df.iterrows():
    plt.plot(timestamps_df.iloc[0], row, linestyle='-', marker='o', alpha=0.5)

plt.show()
