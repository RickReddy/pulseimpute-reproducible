import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats


fig, ax = plt.subplots()

HCPA = pd.read_csv("hcpa_ppg_MSE.csv").to_numpy().flatten()
HCPYA = pd.read_csv("hcpya_ppg_MSE.csv").to_numpy().flatten()
NKI = pd.read_csv("nki_ppg_MSE.csv").to_numpy().flatten()
eegfmri_VU = pd.read_csv("eegfmri_VU_ppg_MSE.csv").to_numpy().flatten()
eegfmri_NIH = pd.read_csv("eegfmri_NIH_ppg_MSE.csv").to_numpy().flatten()

ax.boxplot([HCPA, HCPYA, NKI, eegfmri_VU, eegfmri_NIH], labels = ["HCPA", "HCPYA", "NKI", "eegfmri_VU", "eegfmri_NIH"])
plt.show()