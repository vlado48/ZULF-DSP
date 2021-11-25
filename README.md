## ZULF DSP
The aim of this module is to provide well rounded digital signal processing that specifically adressess needs of Zero-to-Ultralow field NMR time signals. Single object constructor call acts as entire DSP pipeline. Default values provide robust parameters that give decent results in most cases of ZULF NMR data. Methods then provide options for quick visualization and parameter iteration. Spectrum object stores the data (time and frequency domains) as they were at each postprocessing stage for later investigation.
	
## Dependencies
Project will require:
* Python 3.6
* Spectrum 0.8.0 (Library)
* nmrglue 0.9.0 (Library)
* Python standard libraries
	
## Setup
Project is not packaged. Use requires the files to either be in current working directory or in any PYTHONPATH folder. Import ZULFDSP module in your code.

## The Postprocessing Pipeline
#### Raw data accessing
Constructor will accept: 
- absolute path to local data file
- initialized np.array 
- will prompt a dialog to select file manually

#### Removal of corrupted datapoints
More ofte than no optical magnetometer recuperation period affects first few datapoints which have to be removed. Or a pulse (figure) has been used, rendering first dozens of 
datapoints useless.

<img src="https://user-images.githubusercontent.com/34378363/143435344-ba9baa5d-ab70-4cfd-853f-4396d3f812c0.png" width=50% height=50%>

#### Signal Debasing
Signal can be decomposed into Noise + Signal + Bacground. Bacground comes from magnetometer detecting static magnetic field change (loss of polarization of sample). High order polynomial is fitted using linear regression and substracted from the signal.
<img src="https://user-images.githubusercontent.com/34378363/143436140-0d2f02d5-752c-45dd-88e7-c123e4e71941.png" width=50% height=50%>
#### Backward Prediction
Corrupted points that were removed create a linear phase shift across the spectra. Using inverted autoregressive coeffecients we create backward predicting model and fill the missing points with spectrally identical signal.

<img src="https://user-images.githubusercontent.com/34378363/143437184-41e09007-f6ba-4aa5-a1d7-419ea9939d82.png" width=50% height=50%>

#### Filtering
Simple filter aimed to reduce the presence of frequencies coming from the power grid (50Hz/60Hz and harmonics)

<img src="https://user-images.githubusercontent.com/34378363/143436307-eec61153-25ef-4373-b395-5d2de2904ff9.png" width=50% height=50%>

#### Zero Padding
To increase freqency resolution of spectra we can artificially increase measurement time by adding zeros at the end. We can see that by having smaller frequency bins, spectra provides additional details:

<img src="https://user-images.githubusercontent.com/34378363/143437385-4ac231c8-87c1-41e9-b814-a2e9bf40144f.png" width=50% height=50%>

#### Apodization
Signal is multiplied by decaying exponential. Apodization serves primarily two purposes:
- Mitigate artefacts that would result from sudden jump between last point of signal and first zero in zero padding
- Improve SNR by decreasing amplitude of later part of signal where SNR is lower due to sample's loss of polarization.

<img src="https://user-images.githubusercontent.com/34378363/143436781-9bf0bd0a-e7d6-4059-bfbc-e5ed8edcda72.png" width=50% height=50%>
SNR of spectra is imroved at the cost of total amplitude

<img src="https://user-images.githubusercontent.com/34378363/143437055-6194a549-74d0-4a27-8ba8-c280b812e18f.png" width=50% height=50%>

#### Phase correction
Phase of the spectra usually needs additional correction:
- Either can be set by a constant value during constructor call
- Prompt a interactive plot

<img src="https://user-images.githubusercontent.com/34378363/143437457-253e90f0-d968-4fc5-81ca-ce578dd88f47.png" width=50% height=50%>

## Methods
#### Spectrum.plot()
Allows for plotting of selected postprocessing stage, time or frequency domain and given range.

#### Spectrum.iterate()
Is used to iterate a postprocessing parameter and display the comparison on selected data mode and range. It is possible to iterate over two parameters, showing spectra with all the permutations of the parameters. (Use cautiously). On example we can see what effect on final spectra would have a different amount of starting points removed:
<img src="https://user-images.githubusercontent.com/34378363/143439563-c06f4f98-46fe-4bdf-ba2a-810bda1933e9.png" width=50% height=50%>



