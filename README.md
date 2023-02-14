# Non-Markovian-Noise-Project
Python script and Zurich Instruments drivers used for the [non-Markovian noise project](https://arxiv.org/abs/2210.01388)

## Contents

- configuration.py: Initializes Zurich Instruments and peripherals (LO's, attenuators, etc.) for measurement 
- script.py: Contains scripts for executing basic qubit characterizations and project-specific measurements. Experimental parameters are defined in a command table and saved along with data in csv format for later reference.
- comTableFuncs.py: Contains functions that set up and load the command table used by the HDAWG during measurement
- experiment_funcs.py: Contains functions necessary for measurement such as automatic RF mixer calibration, setting up pulsed (Rabi, Ramsey, etc.) and spectroscopy experiments
- HDAWG.py: API driver for Zurich Instruments Arbitrary Waveform Generator (HDAWG)
- UHFQA.py: API driver for Zurich Instruments Quantum Analyzer (UHFQA)
- plot_functions.py: Contains functions for analyzing, fitting, and plotting of data using Seaborn python package
- VISA_drivers (folder): Contains API's for several peripheral instruments such as RF LO generators and spectrum analyzers

## Requirements

### Hardware
- UHFQA
- HDAWG
- Digital Attenuator
- Digital Phase shifter for homodyne measurement
- RF Local Oscillators

