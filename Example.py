import ZULFDSP as zulf
import matplotlib.pyplot as plt

example = zulf.Spectrum(data_src = "ExampleData.dat",
                     ms_to_cut = 100,
                     filter = False, 
                     predict = False,
                     prediction_scan = 1000,
                     prediction_order = 100,
                     prediction_fill = 100,
                     phase_correct = True,
                     phase_shift = 0
                     )


#%%
example.plot("phase_corrected", "ftr")

#%%
print(dir(example))
#%%
example.iterate("phase_corrected", "ftr", "all", "phase_shift", (10, 100))
                                                                      
#%% Step by step postprocessing shown using spectrum.plot method.

# List of all postprocessing stages that can be plotted.
stages = ['opened', 'cut', 'debased', 'predicted',  'zerofilled',
          'apodized', 'phase_corrected']

# Iterate over each stage and plot area of interest of each data mode.
for stage in stages:
    plt.figure()
    plt.suptitle(stage)
    # Time domain
    plt.subplot(1, 3, 1)
    example.plot(stage,'time', 'all')
    # Frequency domain - magnitude
    plt.subplot(1, 3, 2)
    example.plot(stage,'ftm', (5, 50))
    # Frequency domain - real part
    plt.subplot(1, 3, 3)
    example.plot(stage,'ftr', (5, 50))

