import nmr_129_main_rework_postprocessingobjectss as zulf
import matplotlib.pyplot as plt
import numpy as np
import csv

names = [' N-15 UREA DMSO 2048  avg_avg1.000000',
         '15N pyridine 16avg',
         'CS methanol 5.000000 uA_avg',
         'FA try_21000.000_mag_avg',
         'Me_Pyrid_05pi_100s_pol_avg',
         'N-15 pyridine 256 avg manually',
         'Sabre_pyridine2_avg',
         'TMP 64 avg with x pulse_avg1']

#%%
sample = 7
dmmp = zulf.Spectrum(data_src = f'C:/Users/Fotonika/Desktop/ZULF DSP/\
pracafitting/data/{names[sample]}.dat',
                     ms_to_cut = 100,
                     filter = False, 
                     predict = False,
                     prediction_scan = 1000,
                     prediction_order = 100,
                     prediction_fill = 100,
                     phase_correct = True,
                     phase_shift = "manual"
                     )



#%% Step by step postprocessing shown using spectrum.plot method

stages = ['opened', 'cut', 'debased', 'predicted',  'zerofilled',
          'apodized']
for stage in stages:
    plt.figure()
    plt.suptitle(stage)
    plt.subplot(1, 3, 1)
    dmmp.plot(stage,'time', 'all')
    plt.subplot(1, 3, 2)
    dmmp.plot(stage,'ftm', (260, 310))
    plt.subplot(1, 3, 3)
    dmmp.plot(stage,'ftr', (260, 310))

