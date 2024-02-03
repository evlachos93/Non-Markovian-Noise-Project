#!/usr/bin/env python
# coding: utf-8

# ## author: Evangelos Vlachos <evlachos@usc.edu>

# This file contains scripts for basic qubit charaterization and measurement for the Non-Markovian Project. The including exerimental sequences are
# - Resonator spectroscopy
# - Qubit spectroscopy
# - Rabi drive
# - Ramsey and Echo Interferometry (including corresponding statistics gathering)
# - Single Shot
# - T1 (including corresponding statistics gathering)
# 
#
# Required instruments
# - HDAWG and UHFQA
# - LO MW sources for both qubit driving and qubit readout
# - Frequency up and down conversion units
# - Control PC, USB or LAN connection
#
# Recommanded connections for HDAWG and UHFQA
# - Ref clc out of UHFQA connected to Ref clc in of HDAWG
# - Trigger output 1 of HDAWG to Ref/Trigger 1 of UHFQA
# - Enable DIO connection if it is necessary
#

# ## Import Modules

# In[4]:


#  ---------- keyboard shortcuts in 'Help' tap ------------
# restart kernel when somthing changed in subfunctions

# from VISAdrivers import sa_api as sa
import time
import importlib.util
import json
import sys, os
import LO845m as LO
import numpy as np
import UHFQA as qa
import HDAWG as hd
import experiment_funcs as expf
import matplotlib.pyplot as plt
import csv
import glob
import scipy as scy
import plot_functions as pf
import h5py
from VISAdrivers.LabBrick_LMS_Wrapper import LabBrick_Synthesizer
from VISAdrivers.sa_api import *
import utils as ut
pi = np.pi
# from VISAdrivers.vaunix_attenuator_wrapper import VaunixAttenuator

meas_device = "CandleQubit_5"

'''-----------------------------------------------------spectroscopy------------------------------------------------------'''

iteration_spec = ut.get_latest_file('spectroscopy')

# Define experimental options for spectroscopy
options_spec = {
    'frequencies':      np.arange(start=3.0,stop=3.4,step=10e-5), # frequencies are in GHz
    'nAverages':        1024,
    'setup':            0,
    'qubit_drive_amp':     200e-3,
    'readout_drive_amp':     0.7,
    'cav_resp_time':        0.25e-6,
    'integration_length':   2.3e-6,
    'AC_pars':              [0.0,0]
    }

# Run experiment
p_data,I,Q = expf.spectroscopy(daq,awg,qubitLO=qubitLO,**options_spec)
# Plot data
pf.spec_plot(freq=options_spec['frequencies'],I=I,Q=Q,qubit_drive_amp=options_spec['qubit_drive_amp'])

# Save data
exp_pars = options_spec
ut.save_data([I,Q],meas_device,exp_pars,'spectroscopy',iteration_spec)

iteration_spec += 1

'''----------------------------------------------------------Rabi---------------------------------------------------------'''

qubit_drive_amp = 0.2

iteration_rabi = ut.get_latest_file('rabi')

options_rabi = {
    'sampling_rate':        2.4e9,
    'qubitDriveFreq':       3.3313e9,
    'integration_length':   2.3e-6,
    'cav_resp_time':        0.25e-6,
    'nAverages':            128,
    'stepSize':             6e-9,
    'Tmax':                 0.6e-6,
    'amplitude_hd':         qubit_drive_amp,
    'sequence':             'rabi',
    'measPeriod':           600e-6,
    }

# Run experiment
t,I,Q,nPoints = expf.pulse(daq,awg,**options_rabi)

# fit & plot data
fitted_pars,error = pf.fit_data(x_vector=t,y_vector=I,dt=t[-1]/nPoints,**options_rabi)
pf.plot_data(awg,x_vector=t,y_vector=I,fitted_pars=fitted_pars,**options_rabi,iteration=iteration_rabi)

# get pi pulse and threshold for active reset
pi_pulse = np.round(1/2*fitted_pars[1])
threshold = round(np.mean(I)*2**12)

# save data
ut.save_data([t,I],meas_device,options_rabi,'rabi',iteration_rabi)

iteration_rabi += 1

'''---------------------------------------Single Shot-------------------------------------------'''

options_single_shot = {
    'nAverages':        2**10,
    'setup':            0,
    'pi2Width':         1/2*pi_pulse*1e-9,
    'measPeriod':       600e-6,
    'qubit_drive_amp':     A_d,
    'cav_resp_time':        0.25e-6,
    'integration_length':   2.3e-6,
    'AC_pars':              [options_rabi['AC_pars'][0],0],
    'rr_IF':            30e6
    }

data_OFF, data_pi = expf.single_shot(daq,awg,**options_single_shot)

#make 2D histogram
pf.plot_single_shot(data_OFF, data_pi)

'''---------------------------------------------------------Ramsey---------------------------------------------------------'''

iteration_ramsey = ut.get_latest_file('ramsey')

detun = 0e6

options_ramsey = {
    'sampling_rate':    1.2e9,
    'nAverages':        256,
    'Tmax':             25e-6,
    'stepSize':         100e-9,
    'prePulseLength':   1500e-9,
    'postPulseLength':  200e-9,
    'integration_length':   2.3e-6,
    'cav_resp_time':    options_rabi['cav_resp_time'],
    'amplitude_hd':     A_d,
    'active_reset':     True,
    'threshold':        threshold,
    'sequence':         'ramsey',
    'measPeriod':       500e-6,
    'qubitDriveFreq':   options_rabi['qubitDriveFreq']+detun,
    'sweep':            0,
    'pi2Width':         1/2*pi_pulse*1e-9,
    'AC_pars':          [0.3,0],
    'AC_freq':          options_rabi['AC_freq'],
    'RT_pars':          [0,0,0],
    'rr_IF':            30e6
    }

qubitLO.set_freq(options_ramsey['qubitDriveFreq']/1e9)

t,I,Q,nPoints = expf.pulse(daq,awg,setup=[0,0,0],**options_ramsey)

# plot data
data = I
fitted_pars,error = pf.fit_data(x_vector=t,y_vector=data,dt=t[-1]/nPoints,**options_ramsey)
pf.plot_data(awg,x_vector=t,y_vector=data,fitted_pars=fitted_pars,**options_ramsey,iteration=iteration_ramsey,plot_mode=0)

# save data
ut.save_data([t,I,Q],meas_device,options_ramsey,'ramsey',iteration_ramsey)

iteration_ramsey += 1

'''---------------------------------------------------------Echo---------------------------------------------------------'''

iteration_echo = ut.get_latest_file('echo')

options_echo = {
    'sampling_rate':    1.2e9,
    'nAverages':        256,
    'Tmax':             30e-6,
    'stepSize':         100e-9,
    'integration_length': 2.3e-6,
    'prePulseLength':   1500e-9,
    'postPulseLength':  200e-9,
    'cav_resp_time':    options_rabi['cav_resp_time'],
    'amplitude_hd':     A_d,
    'sequence':         'echo',
    'measPeriod':       300e-6,
    'qubitDriveFreq':   options_rabi['qubitDriveFreq']+detun,
    'sweep':            0,
    'pi2Width':         1/2*pi_pulse*1e-9,
    'AC_pars':          [options_rabi['AC_pars'][0],0],
    'AC_freq':          options_rabi['AC_freq'],
    'RT_pars':          [0,0,0],
    'rr_IF':            30e6
}

qubitLO.set_freq(options_echo['qubitDriveFreq']/1e9)

t,I,Q,nPoints = expf.pulse(daq,awg,setup=[0,0,0],**options_echo)
# plot data
data = I
fitted_pars,error = pf.fit_data(x_vector=t,y_vector=data,dt=t[-1]/nPoints,**options_echo)
pf.plot_data(awg,x_vector=t,y_vector=data,fitted_pars=fitted_pars,**options_echo,iteration=iteration_echo,plot_mode=0)

# save data
ut.save_data([t,I,Q],meas_device,options_echo,'echo',iteration_echo)

iteration_echo += 1

'''---------------------------------------------------------T1---------------------------------------------------------'''

iteration_T1 = ut.get_latest_file('T1')

options_T1 = {
    'sampling_rate':    1.2e9,
    'nAverages':        128,
    'Tmax':             100e-6,
    'stepSize':         1000e-9,
    'integration_length': 2.3e-6,
    'cav_resp_time':    options_rabi['cav_resp_time'],
    'amplitude_hd':     A_d,
    'sequence':         'T1',
    'measPeriod':       600e-6,
    'qubitDriveFreq':   options_rabi['qubitDriveFreq']+detun,
    'sweep':            0,
    'pi2Width':         1/2*pi_pulse*1e-9,
    'AC_pars':          options_rabi['AC_pars'],
    'RT_pars':          [0,0,0],
    'rr_IF':            30e6
}

qubitLO.set_freq(options_T1['qubitDriveFreq']/1e9)

t,I,Q,nPoints = expf.pulse(daq,awg,setup=[0,0,0],**options_T1)
# plot data
data = Q
fitted_pars,error = pf.fit_data(x_vector=t,y_vector=data,dt=t[-1]/nPoints,**options_T1)
pf.plot_data(awg,x_vector=t,y_vector=data,fitted_pars=fitted_pars,**options_T1,iteration=iteration_T1,plot_mode=0)


# save data
ut.save_data([t,I,Q],meas_device,options_T1,'T1',iteration_T1)

iteration_T1 += 1

