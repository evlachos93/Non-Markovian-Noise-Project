#!/usr/bin/env python
# coding: utf-8


## Contains functions used to setup and run experiments using HDAWG and UHFQA

# # Import Modules

import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import UHFQA as qa
import HDAWG as hd
#import USB6501 as usb
import zhinst.utils as ziut
import comTablefuncs as ctfuncs
import json
import csv
# import keyboard as kb
import os
from plot_functions import Volt2dBm,Watt2dBm
import itertools
from VISAdrivers.sa_api import *
import collections
# from PyTektronixScope import PyTektronixScope
import seaborn as sns; sns.set() # styling
sns.set_style('ticks')

def snr(sa,fc,thres):
    """
    Computes the SNR of a signal at a particular frequency

    Args:
        sa (class): Spectrum analyzer instrument handler
        fc (float): Frequency of interest
        thres (float): Sensitivity of spectrum analyzer
    """
    # configure

    sa_config_acquisition(device = sa, detector = SA_AVERAGE, scale = SA_LOG_SCALE)
    sa_config_center_span(sa, fc, 0.5e6)
    sa_config_level(sa, thres)
    sa_config_gain_atten(sa, SA_AUTO_ATTEN, SA_AUTO_GAIN, True)
    sa_config_sweep_coupling(device = sa, rbw = 1e3, vbw = 1e3, reject=0)

    # Initialize
    sa_initiate(sa, SA_SWEEPING, 0)
    query = sa_query_sweep_info(sa)
    sweep_length = query["sweep_length"]
    start_freq = query["start_freq"]
    bin_size = query["bin_size"]

    freqs = np.array([start_freq + i * bin_size for i in range(sweep_length)],dtype=float)

    signal = sa_get_sweep_64f(sa)['max']
    plt.plot(1e-9*freqs,signal)
    plt.xticks(np.linspace(min(1e-9*freqs), max(1e-9*freqs),5))
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Power (dBm)')
    plt.show()

    max_ind = np.argmax(signal)
    max_val = np.max(signal)
    mask = np.logical_or (freqs < freqs[max_ind]-10e3, freqs > freqs[max_ind]+10e3)
    noisetemp = signal[mask]
    avg_noise = np.mean(noisetemp)
    snr = max_val-avg_noise


    print("SNR: %.1f\nNoise Floor: %.1f dBm"%(snr,avg_noise))

def lo_isol(sa,inst,fc,mixer='qubit',amp=0.2,plot=1):
    """
    Gets power isolation in units of dB between pulse ON/OFF

    Args:
        sa (class): Spectrum analyzer handle
        inst (class): Zurich Instrument handle
        fc (float): Frequency of LO in Hz
        mixer (str, optional): Which mixer to investigate. Defaults to 'qubit'.
        amp (float, optional): Amplitude of ON pulse in Volts. Defaults to 0.2.
        plot (int, optional): If true, plots spectrum analyzer data. Defaults to 1.

    Returns:
        OFF_power,ON_power: Power when pulse is OFF/ON
    """
    # configure
    sa_config_acquisition(device = sa, detector = SA_AVERAGE, scale = SA_LOG_SCALE)
    sa_config_center_span(sa, fc, 0.5e6)
    sa_config_gain_atten(sa, SA_AUTO_ATTEN, SA_AUTO_GAIN, True)
    sa_config_sweep_coupling(device = sa, rbw = 1e3, vbw = 1e3, reject=0)


    if mixer == 'qubit':
        # qubit mixer
        offset_qubit_ch1 = inst.get('/dev8233/sigouts/0/offset')['dev8233']['sigouts']['0']['offset']['value']
        sa_config_level(sa, -50)
        sa_initiate(sa, SA_SWEEPING, 0)
        query = sa_query_sweep_info(sa)
        sweep_length = query["sweep_length"]
        start_freq = query["start_freq"]
        bin_size = query["bin_size"]
        freqs = np.array([start_freq + i * bin_size for i in range(sweep_length)],dtype=float)
        #get OFF power (leakage)
        signal_OFF = sa_get_sweep_64f(sa)['max']
        OFF_power = np.max(signal_OFF)
    
        #get ON power
        sa_config_level(sa, 0)
        sa_initiate(sa, SA_SWEEPING, 0)
        inst.setDouble('/dev8233/sigouts/0/offset', amp)
        inst.sync()
        signal_ON = sa_get_sweep_64f(sa)['max']
        ON_power = np.max(signal_ON)
        inst.setDouble('/dev8233/sigouts/0/offset', offset_qubit_ch1)
        inst.sync()
       
    elif mixer == 'ac':
        # AC stark mixer
        #get OFF power (leakage)
        offset_ac_stark_ch1 = inst.get('/dev8233/sigouts/1/offset')['dev8233']['sigouts']['1']['offset']['value']
        sa_config_level(sa, -50)
        sa_initiate(sa, SA_SWEEPING, 0)
        query = sa_query_sweep_info(sa)
        sweep_length = query["sweep_length"]
        start_freq = query["start_freq"]
        bin_size = query["bin_size"]
        freqs = np.array([start_freq + i * bin_size for i in range(sweep_length)],dtype=float)
        #get OFF power (leakage)
        signal_OFF = sa_get_sweep_64f(sa)['max']
        OFF_power = np.max(signal_OFF)
        #get ON power
        sa_config_level(sa, 0)
        sa_initiate(sa, SA_SWEEPING, 0)
        inst.setDouble('/dev8233/sigouts/1/offset', amp)
        inst.sync()
        signal_ON = sa_get_sweep_64f(sa)['max']
        ON_power = np.max(signal_ON)
        inst.setDouble('/dev8233/sigouts/1/offset', offset_ac_stark_ch1)
        
    elif mixer == 'readout':
        # readout mixer
        offset_readout_ch1 = inst.get('/dev2528/sigouts/0/offset')['dev2528']['sigouts']['0']['offset']['value']
        #get OFF power (leakage)
        sa_config_level(sa, -50)
        sa_initiate(sa, SA_SWEEPING, 0)
        query = sa_query_sweep_info(sa)
        sweep_length = query["sweep_length"]
        start_freq = query["start_freq"]
        bin_size = query["bin_size"]
        freqs = np.array([start_freq + i * bin_size for i in range(sweep_length)],dtype=float)
        #get OFF power (leakage)
        signal_OFF = sa_get_sweep_64f(sa)['max']
        OFF_power = np.max(signal_OFF)
        #get ON power
        sa_config_level(sa, 0)
        sa_initiate(sa, SA_SWEEPING, 0)
        inst.set('/dev2528/sigouts/0/offset', amp)
        inst.sync()
        signal_ON = sa_get_sweep_64f(sa)['max']
        ON_power = np.max(signal_ON)
        inst.set('/dev2528/sigouts/0/offset', offset_readout_ch1)
        inst.sync()
        
    if plot:
        plt.plot(np.around(freqs,5),signal_ON)
        plt.xticks(np.around(np.linspace(min(freqs), max(freqs),5),5))
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Power (dBm)')
        plt.show()

    print("LO Off Power: %.1f dBm\nLO On Power: %.1f dBm\nLO Isolation: %.1f"%(OFF_power,ON_power,ON_power-OFF_power))

    return OFF_power,ON_power


def mixer_calib(sa,inst,mode,mixer='qubit',fc=3.875e9,amp=0.2):
    """
    DESCRIPTION:
        Optimizes mixer at given frequency

    INPUTS:
        device (class): The instrument (AWG or QA) that controls the I and Q channels of the mixer we want to optimize
        mixer (string): The mixer we want to optimize. Options are "qubit","resonator",and "stark". Defaults to 'qubit'.
        f_c (float): The frequency we want to optimize at. Defaults to 3.875e9.
        mode(string): Coarse or fine tuning
        amp (float): Amplitude of ON Pulse
    """

    if mode == 'coarse':
        span=20e-3
        dV=1e-3
    elif mode == 'fine':
        span=2e-3
        dV=0.1e-3

    # Open device
    if mixer == 'qubit':
        device = 'dev8233'
        channels = [0,2]
    elif mixer == 'ac':
        device = 'dev8233'
        channels = [1,3]
    elif mixer == 'readout':
        device = 'dev2528'
        channels = [0,1]

    if str(inst) == 'awg':
        device = 'dev8233'
    elif str(inst) == 'daq':
        device = 'dev2528'

    vStart = np.zeros(2)
    for i in range(len(vStart)):
        vStart[i] = inst.get('/%s/sigouts/%d/offset'%(device,channels[i]))['%s'%(device)]['sigouts']['%d'%channels[i]]['offset']['value']
    VoltRange1 = np.arange(vStart[0]-span/2,vStart[0]+span/2,dV)
    VoltRange2 = np.arange(vStart[1]-span/2,vStart[1]+span/2,dV)

    OFF_power1 = np.zeros(len(VoltRange1))
    OFF_power2 = np.zeros(len(VoltRange2))
    # Sweep individual channel voltages and find leakage
    for i in range(len(VoltRange1)):
        inst.setDouble('/%s/sigouts/%d/offset'%(device,channels[0]),VoltRange1[i])
        inst.sync()
        OFF_power1[i],ON_power = lo_isol(sa, inst, fc,mixer=mixer,plot=0,calib=1)

    min_ind1 = np.argmin(OFF_power1)
    inst.set('/%s/sigouts/%d/offset'%(device,channels[0]),VoltRange1[min_ind1])
    inst.sync()

    for i in range(len(VoltRange2)):
        inst.setDouble('/%s/sigouts/%d/offset'%(device,channels[1]),VoltRange2[i])
        inst.sync()
        OFF_power2[i],ON_power = lo_isol(sa, inst, fc,mixer=mixer,plot=0,calib=1)

    # find index of voltage corresponding to minimum LO leakage
    # min_ind1 = np.argmin(OFF_power1)
    min_ind2 = np.argmin(OFF_power2)
    # set voltages to optimal values

    inst.set('/%s/sigouts/%d/offset'%(device,channels[1]),VoltRange2[min_ind2])
    inst.sync()

    OFF_power,ON_power = lo_isol(sa, inst, fc,mixer=mixer,amp=amp,plot=1,calib=0)

    fig,ax = plt.subplots(1,2,sharex=False,sharey=True,squeeze=False)
    ax[0][0].plot(VoltRange1*1e3,OFF_power1,'-o',c='r')
    ax[0][0].set_xlabel('Channel %d Voltage (mV)'%(channels[0]))
    ax[0][0].set_ylabel('LO Leakage (dBm)')
    ax[0][1].plot(VoltRange2*1e3,OFF_power2,'-o',c='b')
    ax[0][1].set_xlabel('Channel %d Voltage (mV)'%(channels[1]))
    textstr = 'Ch1 Voltage (mV):%.2f\nCh2 Voltage (mV):%.2f\nOFF Power (dBm): %.1f\nON Power (dBm): %.1f'%(VoltRange1[min_ind1]*1e3,VoltRange2[min_ind2]*1e3,OFF_power,ON_power)
    plt.gcf().text(0.925, 0.45, textstr,fontsize=10,bbox=dict(boxstyle='round,rounding_size=1.25',facecolor='silver',alpha=0.5))
    print('Ch1 Voltage (mV):%.2f\nCh2 Voltage (mV):%.2f\nOFF Power (dBm): %.1f\nON Power (dBm): %.1f'%(VoltRange1[min_ind1]*1e3,VoltRange2[min_ind2]*1e3,OFF_power,ON_power))

def readoutSetup(awg,sequence='spec',readout_pulse_length=1.2e-6,rr_IF=5e6,cav_resp_time=0.5e-6):
    """
    Setups UHFQA's AWG to execute readout

    Args:
        awg (class): UHFQA instrument handle
        sequence (str, optional): Specifies whether to prepare readout for spectroscopy or pulsed experiment. Defaults to 'spec'.
        readout_pulse_length (float, optional): Length of readout pulse. Defaults to 1.2e-6.
        rr_IF (float, optional): IF frequency of readout LO. Defaults to 5e6.
        cav_resp_time (float, optional): Response time of cavity. Defaults to 0.5e-6.
    """
    fs = 450e6
    readout_amp = 0.7
    # prepares QA for readout | loads readout Sequence into QA AWG and prepares QA
    print('-------------Setting up Readout Sequence-------------')
    if sequence =='spec':
        qa.awg_seq_readout(awg,'dev2528', readout_length=readout_pulse_length,rr_IF=rr_IF,nPoints=roundToBase(readout_pulse_length*fs),base_rate=fs,cav_resp_time=cav_resp_time,amplitude_uhf=readout_amp)
        awg.setInt('/dev2528/awgs/0/time',2)
    elif sequence =='pulse':
        qa.awg_seq_readout(awg,'dev2528', readout_length=readout_pulse_length,rr_IF=rr_IF,nPoints=roundToBase(readout_pulse_length*fs),base_rate=fs,cav_resp_time=cav_resp_time,amplitude_uhf=readout_amp)
        awg.setInt('/dev2528/awgs/0/time',2)

def pulsed_spec_setup(daq,awg,nAverages,qubit_drive_amp,qubit_drive_dur=30e-6,result_length=1,integration_length=2e-6,delay=500):
    """
    Sets up pulsed spectroscopy experiment

    Args:
        daq (class): Handle for UHFQA.
        awg (class): Handle for HDAWG.
        nAverages (int): Number of averages used in the experiment. Must be a power of 2.
        qubit_drive_amp (float): Amplitude of the qubit excitation pulse in Volts.
        qubit_drive_dur (float, optional): Duration of qubit excitation pulse in seconds. Defaults to 30e-6.
        result_length (int, optional): Number of sequence points. Defaults to 1.
        integration_length (float, optional): Length of readout pulse in seconds. Defaults to 2e-6.
        delay (int, optional): Delay after UHFQA's AWG receives trigger to start readout. Defaults to 500.
    """
    hd.awg_seq(awg,sequence='qubit spec',fs=0.6e9,nAverages=nAverages,qubit_drive_dur=qubit_drive_dur,amplitude_hd= qubit_drive_amp)
    awg.setInt('/dev8233/awgs/0/time',2) # sets AWG sampling rate to 600 MHz
    qa.config_qa(daq,sequence='spec',integration_length=integration_length,nAverages=1,result_length=result_length,delay=delay)

def single_shot_setup(daq,awg,nAverages=1024,qubit_drive_amp=0.1,fs=0.6e9,result_length=1,integration_length=2e-6,pi2Width=100e-9,measPeriod=400e-6):
    """
    Sets up UHFQA and HDAWG for single shot experiment

    Args:
        daq (class): Handle for UHFQA.
        awg (class): Handle for HDAWG.
        nAverages (int, optional): Number of averages used in the experiment. Has to be a power of 2. Defaults to 1024.
        qubit_drive_amp (float, optional): Amplitude of qubit reset pulse. Defaults to 0.1.
        fs (float, optional): Sampling rate of HDAWG. Defaults to 0.6e9. 
        result_length (int, optional): Number of sequence points. Defaults to 1.
        integration_length (float, optional): Length of readout pulse in seconds. Defaults to 2e-6.
        pi2Width (float, optional): Duration of pipulse. Defaults to 100e-9.
        measPeriod (float, optional): Waiting period between experiments in seconds. Defaults to 400e-6.
    """
    pi2Width = int(pi2Width*fs)
    print('-------------Setting HDAWG sequence-------------')
    hd.awg_seq(awg,sequence='single_shot',fs=fs,nAverages=nAverages,AC_pars=AC_pars,amplitude_hd=qubit_drive_amp,measPeriod=measPeriod,pi2Width=pi2Width)
    awg.setInt('/dev8233/awgs/0/time',2) # sets AWG sampling rate to 600 MHz

def seq_setup(awg,sequence='rabi',nAverages=128,prePulseLength=1500e-9,postPulseLength=1500e-9,nPoints=1024,pulse_length_start=32,fs=2.4e9,nSteps=100,pulse_length_increment=16,Tmax=0.3e-6,amplitude_hd=1,AC_pars=[0.4,0],pi2Width=0,piWidth_Y=0,pipulse_position=20e-9,measPeriod=200e-6,instance=0,sweep_name=0,sweep=0,RT_pars=[0,0],active_reset=False):
    """
    Function Description
    --------------------

    Sets up the right sequence to the AWG along with the corresponding command table

    Parameters
    ----------
    awg : class
        awg to write the sequence to
    'dev8233' : string
        serial number of AWG. Default is dev8233
    sequence : string
        Type of experiment. Available options are 'rabi','ramsey','echo', and 'T1'. The default is 'rabi'.
    nAverages : integer
    nPoints : integer
        Number of points in the waveform. The default is 1024.
    fs : float
        Sampling rate of the AWG. The default is 2.4e9.
    nSteps : integer
        Number of points in the sequence. The default is 100.
    pulse_length_increment : integer
        dt in samples. The default is 16.
    Tmax : float
    active_reset: Boolean
    amplitude_hd : TYPE, optional
        qubit drive amplitude. The default is 1.
    AC_pars : TYPE, optional
        A list describing the amplitude (mu) and standard deviation (sigma) of the AC stark tone ([mu,sigma]).
    pi2Width : TYPE, optional
        pi/2 duration The default is 50e-9.
    measPeriod : TYPE, optional
        waiting interval between measurements. Must be at least 2*T_1. The default is 200e-6.
    RT_pars : TYPE, optional
        A list describing the amplitude  (B_0) and decay time (tau_k) of gen. Markovian noise ([B_0,tau_k]).

    """
    fs_base = 2.4e9
    pi2Width = round(fs * pi2Width)
    piWidth_Y = round(fs*piWidth_Y)
    if AC_pars[0] != 0 or AC_pars[1] != 0:
        nPointsPre = roundToBase(prePulseLength*fs+pi2Width)
        nPointsPost = roundToBase(postPulseLength*fs+pi2Width)
        if nPointsPre > 2048 or nPointsPost > 2048:
            print('Too many points in your AC pre/post pulses!(%d,%d)'%(nPointsPre,nPointsPost))
            sys.exit()
        else:
            pass
    else:
        nPointsPre = 0
        nPointsPost = 0

    # Generate and compile program
    print('-------------Setting HDAWG sequence-------------')
    bt = time.time()
    awg.setInt('/dev8233/awgs/0/time',(int(fs_base/fs-1))) # set sampling rate of AWG to 2.4 GHz
    hd.awg_seq(awg,AC_pars=AC_pars,fs=fs,nPoints=nPoints,pulse_length_increment=pulse_length_increment,nSteps=nSteps,nPointsPre=nPointsPre,pi2Width=pi2Width,pipulse_position=round(fs*pipulse_position),piWidth_Y=piWidth_Y,nPointsPost=nPointsPost,Tmax=Tmax,amplitude_hd=amplitude_hd,nAverages=nAverages,sequence=sequence,measPeriod=measPeriod,RT_pars=RT_pars,active_reset=active_reset)
    et = time.time()
    print('HDAWG compilation duration: %.1f s'%(et-bt))

    # create and upload command table
    ct=ctfuncs.ct_pulse_length(n_wave=nSteps, pulse_length_start=pulse_length_start, pulse_length_increment=pulse_length_increment,AC_amp=AC_pars[0], AC_pars=[nPointsPre,nPointsPost,int(2*pi2Width)], active_reset=active_reset,sequence=sequence)
    awg.setVector("/dev8233/awgs/0/commandtable/data", json.dumps(ct))

    awg.sync()
    # print(flatten(ct))
    return nSteps,ct



def single_shot(daq,awg,cav_resp_time=1e-6,measPeriod=400e-6,integration_length=2.3e-6,AC_pars=[0,0],rr_IF=30e6,pi2Width=100e-9,qubit_drive_amp=1,readout_drive_amp=0.1,setup=0,nAverages=128):
    '''
    DESCRIPTION: Executes single shot experiment so best thresholding value is determined

    '''
    result_length =  2*nAverages
    fsAWG = 600e6
    base_rate = 1.8e9

    readout_pulse_length = integration_length + cav_resp_time + 1e-6

    if not setup:
        single_shot_setup(daq,awg,pi2Width=pi2Width,result_length=result_length,fs=fsAWG,AC_pars=AC_pars,integration_length=integration_length,nAverages=nAverages,qubit_drive_amp=qubit_drive_amp,measPeriod=measPeriod)
        readoutSetup(daq,sequence='spec',readout_pulse_length=readout_pulse_length,cav_resp_time=cav_resp_time)
        time.sleep(0.1)

    sweep_data, paths = qa.create_sweep_data_dict(daq, 'dev2528')
    data_pi = []
    data_OFF = []

    bt = time.time()
    qa.qa_result_reset(daq, 'dev2528')
    hd.enable_awg(awg, 'dev8233',enable=0,awgs=[0])
    qa.config_qa(daq,sequence='single shot',nAverages=1,integration_length=integration_length,result_length=result_length,delay=cav_resp_time)
    daq.sync()

    qa.qa_result_enable(daq, 'dev2528')
    qa.enable_awg(daq, 'dev2528') # start the readout sequence
    hd.enable_awg(awg,'dev8233',enable=1,awgs=[0])

    print('Start measurement')
    data = qa.acquisition_poll(daq, paths, result_length, timeout = 3*nAverages*measPeriod) # transfers data from the QA result to the API for this frequency point
    # seperate OFF/ON data and average
    data_OFF = np.append(data_OFF, [data[paths[0]][k] for k in even(len(data[paths[0]]))])/(integration_length*base_rate)
    data_pi =  np.append(data_pi, [data[paths[0]][k] for k in odd(len(data[paths[0]]))])/(integration_length*base_rate)


    hd.enable_awg(awg, 'dev8233',enable=0,awgs=[0])
    qa.stop_result_unit(daq, 'dev2528', paths)
    qa.enable_awg(daq, 'dev2528', enable = 0)

# ----------------------------------------------------------------------------------
    et = time.time()
    duration = et-bt
    print(f'Measurement time: {duration} s')
#-----------------------------------

    return data_OFF,data_pi

def spectroscopy(daq,awg,qubitLO=0,cav_resp_time=1e-6,integration_length=2e-6,AC_pars=[0,0],qubit_drive_amp=1,readout_drive_amp=0.1,setup=0,nAverages=128,frequencies=np.linspace(3.7,3.95,1001)):
    '''
    DESCRIPTION: Executes qubit spectroscopy.

    '''
    result_length =  2*nAverages
    fsAWG = 600e6
    base_rate = 1.8e9

    readout_pulse_length = integration_length + cav_resp_time + 2e-6
    # daq.setDouble('/dev2528/sigouts/0/amplitudes/0', readout_drive_amp)
    nPointsPre = nPointsPost = roundToBase(500e-9*fsAWG,base=16)
    qubit_drive_dur = roundToBase(30e-6*fsAWG,base=16)
    if not setup:
        pulsed_spec_setup(daq, awg, result_length=result_length,AC_pars=AC_pars,qubit_drive_dur=qubit_drive_dur,integration_length=integration_length,nAverages=nAverages,qubit_drive_amp=qubit_drive_amp,nPointsPre=nPointsPre,nPointsPost=nPointsPost,delay=cav_resp_time)
        readoutSetup(daq, sequence='spec',readout_pulse_length=readout_pulse_length)
        time.sleep(0.1)

    # initialize signal generators and set power


    print('Start measurement')
    sweep_data, paths = qa.create_sweep_data_dict(daq, 'dev2528')
    data_ON = []
    data_OFF = []


    qa.enable_awg(daq, 'dev2528') # start the readout sequence
    bt = time.time()
    j = 0
    for f in frequencies:
        qubitLO.set_freq(f)
        qa.qa_result_reset(daq, 'dev2528')
        qa.qa_result_enable(daq, 'dev2528')
        hd.enable_awg(awg,'dev8233',awgs=[0]) #runs the drive sequence
        data = qa.acquisition_poll(daq, paths, result_length, timeout = 60) # transfers data from the QA result to the API for this frequency point
        # seperate OFF/ON data and average
        data_OFF = np.append(data_OFF, np.mean([data[paths[0]][k] for k in even(len(data[paths[0]]))]))
        data_ON =  np.append(data_ON, np.mean([data[paths[0]][k] for k in odd(len(data[paths[0]]))]))

        sys.stdout.write('\r')
        sys.stdout.write(f'progress:{int((j+1)/len(frequencies)*100)}%')
        sys.stdout.flush()
        j = j + 1

    data = (data_ON-data_OFF)/(integration_length*base_rate)
    I_data= data.real
    Q_data = data.imag

    power_data = np.abs(I_data*I_data.conjugate()+Q_data*Q_data.conjugate())

    hd.enable_awg(awg, 'dev8233',enable=0,awgs=[0])
    qa.stop_result_unit(daq, 'dev2528', paths)
    qa.enable_awg(daq, 'dev2528', enable = 0)

# ----------------------------------------------------------------------------------
    et = time.time()
    duration = et-bt
    print(f'Measurement time: {duration} s')
#-----------------------------------

    return power_data,I_data,Q_data

def pulse(daq,awg,setup=[0,0,0],Tmax=0.3e-6,nSteps=61,prePulseLength=1500e-9,postPulseLength=1500e-9,nAverages=128,amplitude_hd=1,
          sequence='rabi',AC_pars=[0,0],stepSize=2e-9, RT_pars=[0,0,0],cav_resp_time=0.5e-6,piWidth_Y=0,AC_freq=5e-9,
          pipulse_position=20e-9,integration_length=2.3e-6,qubitDriveFreq=3.8135e9,pi2Width=0,rr_IF = 30e6,sampling_rate=1.2e9,
          measPeriod=300e-6,sweep=0,sweep_name='sweep_001',instance=0,active_reset=False,threshold=500e-3,noise_instance=np.zeros(10)):

    '''
    DESCRIPTION:            Runs a single pulsed experiment (Rabi,Ramsey,T1,Echo)
    -----------------------------------------------------------------------------------------------------------------
    setup[0]:                   If setup[0]=0, the right seqc programs are loaded into the HDAWG. If setup[0]=2, the noise waveforms are substituted and there is no compilation (used for sweeps or statistics)
    setup[1]:                   If setup[1]=0, the right seqc programs are loaded into the QA AWG for readout
    setup[2]:                   If setup[2]=0, QA is configured
    Tmax:                       max length of drive (rabi) or pi2 pulse separation
    amplitude_hd:               amplitude of qubit drive channel
    sequence:                   Which experiment to perform (see description)
    pi2Width:                   Length of pi2 pulse in seconds
    instance:                   Which instance of telegraph noise to use. Used for sweeps
    rr_IF:                      The IF of the readout mixer
    'pipulse_position':         Where to insert the pipulse (only applicable for echo with telegraph noise). A higher number means the pipulse is applied sooner
    cav_resp_time:              The time it takes for the cavity to ring up/down. Since we are using square integration pulse, we don't want to include the edges of the pulse
    integration_length:         How long the QA integrates for. The readout pulse is 2 microseconds longer than integration+cavity_response
    '''
    fs = sampling_rate
    base_rate = 1.8e9       # sampling rate of QA (cannot be changed in standard mode)
    readout_pulse_length = integration_length + cav_resp_time + 1e-6

    if sequence == 'echo' and RT_pars[0] != 0:
        if int(fs*stepSize) < 64:
            print('Error: The minimum step size is 64 pts')
            sys.exit()
        cores = [1,0]
    else:
        cores = [0]

    # stops AWGs and reset the QA
    hd.enable_awg(awg,'dev8233',enable=0,awgs=cores)
    qa.enable_awg(daq, 'dev2528',enable=0)
    qa.qa_result_reset(daq, 'dev2528')

    if setup[0] == 0:
        nPoints,nSteps,pulse_length_increment,pulse_length_start = calc_nSteps(sequence=sequence,fsAWG=fs,piWidth_Y=piWidth_Y,stepSize=stepSize,Tmax=Tmax,RT_pars=RT_pars)
        if sequence == 'ramsey' or sequence == 'echo' or sequence == 'T1':
            create_wfm_file(AC_pars=AC_pars, RT_pars=RT_pars,sequence=sequence, nPoints=nPoints, sweep=sweep, sweep_name=sweep_name, Tmax=Tmax, instance=instance)
        nSteps,ct = seq_setup(awg,sequence=sequence,piWidth_Y=piWidth_Y,pipulse_position=pipulse_position,nSteps=nSteps,nPoints=nPoints,fs=fs,pulse_length_start=pulse_length_start,pulse_length_increment=pulse_length_increment,instance=instance,sweep_name=sweep_name,amplitude_hd=amplitude_hd,nAverages=nAverages,pi2Width=pi2Width,prePulseLength=prePulseLength,postPulseLength=postPulseLength,Tmax=Tmax,AC_pars=AC_pars,RT_pars=RT_pars,measPeriod=measPeriod,sweep=sweep,active_reset=active_reset)
        print('setup complete')
    elif setup[0] == 2:
        bt = time.time()
        # replace waveforms, don't recompile program
        # noise_instance = pull_wfm(sweep_name=sweep_name, RT_pars=RT_pars, instance=instance)
        nPoints,nSteps,pulse_length_increment,pulse_length_start = calc_nSteps(sequence=sequence,fsAWG=fs,piWidth_Y=piWidth_Y,stepSize=stepSize,Tmax=Tmax,RT_pars=RT_pars)
        if AC_pars[0] != 0:
            white_noise = np.random.normal(AC_pars[0], AC_pars[1], nPoints)
            waveforms_native = ziut.convert_awg_waveform(wave1=noise_instance,wave2=white_noise)
        else:
            waveforms_native = ziut.convert_awg_waveform(wave1=noise_instance)
        path = '/dev8233/awgs/0/waveform/waves/0'
        awg.setVector(path,waveforms_native)
        et = time.time()
        print('replacing waveforms took: %.1f ms'%(1e3*(et-bt)))

    time.sleep(1)

    if setup[1] == 0:
        readoutSetup(daq,readout_pulse_length=readout_pulse_length,sequence='pulse',rr_IF=rr_IF,cav_resp_time=cav_resp_time)
    if setup[2] == 0:
        qa.config_qa(daq,sequence='pulse',nAverages=nAverages,rr_IF=rr_IF,integration_length=integration_length,result_length=nSteps,delay=cav_resp_time)
        daq.sync()
    if active_reset == True:
        setup_active_reset(awg, daq,threshold=threshold)

    # Determine whether command table is used for error checking later on
    if AC_pars[0] != 0:
        use_ct = 1
    elif AC_pars[0] == 0 and sequence == 'rabi':
        use_ct = 1
    else:
        use_ct = 0
    print('Estimated Measurement Time (without active reset): %d sec'%(calc_timeout(nAverages, measPeriod, stepSize, nSteps)))
    # Checks whether the right command table is used
    ct_awg = json.loads(daq.get("/dev8233/awgs/0/commandtable/data",flat=True)["/dev8233/awgs/0/commandtable/data"][0]['vector'])
    if setup[0] == 0 and use_ct == 1:
        if ct_awg != ct:
            print('Error! Invalid Command Table used for Measurement\nCommand Table Sent to AWG\n\n%s\n\nCommand Table in AWG\n\n%s'%(ct,ct_awg))
            sys.exit()

    result_length = nSteps
    timeout = 2*nSteps*measPeriod*nAverages
    sweep_data, paths = qa.create_sweep_data_dict(daq, 'dev2528')

    qa.enable_awg(daq, 'dev2528',enable=1) # start the readout sequence

    qa.qa_result_enable(daq, 'dev2528')

    str_meas = time.time()
    hd.enable_awg(awg,'dev8233',enable=1,awgs=cores) #runs the drive sequence
    data = qa.acquisition_poll(daq, paths, num_samples = result_length, timeout = timeout)

    for path, samples in data.items():
        sweep_data[path] = np.append(sweep_data[path], samples)

    qa.stop_result_unit(daq, 'dev2528', paths)
    hd.enable_awg(awg, 'dev8233', enable = 0,awgs=cores)
    qa.enable_awg(daq, 'dev2528', enable = 0)
    end_meas = time.time()
    print('\nmeasurement duration: %.1f s' %(end_meas-str_meas))

    data = sweep_data[paths[0]][0:result_length]/(integration_length*base_rate)
    I = data.real
    Q = data.imag

    #Generate time array points
    t = np.zeros(nSteps)
    if use_ct == 1:
        # ct_awg = json.loads(daq.get("/dev8233/awgs/0/commandtable/data",flat=True)["/dev8233/awgs/0/commandtable/data"][0]['vector']) #
        for i in range(nSteps):
            t[i] = ct_awg['table'][i]['waveform']['length']/fs
    else:
        t = np.linspace(pulse_length_start/fs,Tmax,nSteps)

    if sequence=='echo':
        t = 2*t

    return t,I,Q,nSteps

def adapt_stepsize(B0):
    # need at least 10 points per period
    return 1/(10*2*B0*25*1e6)

def rabi_ramsey(daq,awg,qubitLO,qubitDriveFreq=3.8e9,AC_pars=[0,0],plot=1):
    '''---------------------------------------Do Rabi, calibrate pi-pulse, then do Ramsey----------------------------------------------'''

    optionsRabi= {
        'nAverages':        128,
        'Tmax':             0.2e-6,
        'amplitude_hd':     1.0,
        'sequence':         'rabi',
        'channel':          0,
        'measPeriod':       200e-6,
        'qubitDriveFreq':   qubitDriveFreq,
        'AC_pars':          AC_pars
        }

    t,ch1Data,ch2Data,nPoints = pulse(daq,awg,qubitLO,setup=[0,0,0],**optionsRabi)
    pi_pulse,error = pf.pulse_plot1d(sequence='rabi',dt=optionsRabi['Tmax']*1e6/nPoints,qubitDriveFreq=optionsRabi['qubitDriveFreq'],amplitude_hd=optionsRabi['amplitude_hd'],x_vector=t, y_vector=ch1Data,fitting=1,AC_pars=optionsRabi['AC_pars'],plot=0)

    optionsRamsey = {
        'nAverages':        128,
        'Tmax':             10e-6,
        'stepSize':         100e-9,
        'pi2Width':         1/2*pi_pulse*1e-9,
        'amplitude_hd':     optionsRabi['amplitude_hd'],
        'sequence':         'ramsey',
        'channel':          0,
        'measPeriod':       200e-6,
        'qubitDriveFreq':   optionsRabi['qubitDriveFreq'],
        'AC_pars':          optionsRabi['AC_pars'],
        'RT_pars':          [0,0]

        }


    t,ch1Data,ch2Data,nPoints = pulse(daq,awg,qubitLO,setup=[0,0,0],**optionsRamsey)
    # plot data
    detuning,T_phi,error = pf.pulse_plot1d(sequence='ramsey',dt=optionsRamsey['Tmax']*1e6/nPoints,qubitDriveFreq=optionsRamsey['qubitDriveFreq'],amplitude_hd=optionsRamsey['amplitude_hd'],x_vector=t, y_vector=ch1Data,fitting=1,AC_pars=optionsRamsey['AC_pars'],pi2Width=optionsRamsey['pi2Width'],RT_pars=optionsRamsey['RT_pars'],plot=plot)

    return detuning,T_phi,error

def calc_nSteps(sequence='ramsey',fsAWG=1.2e9,stepSize=10e-9,Tmax=5e-6):
    """
    Calculates the number of steps in the sequence and the number of points in the waveform in AWG units

    Args:
        sequence (str, optional): Type of sequence. Defaults to 'ramsey'.
        fsAWG (float, optional): Sampling rate of HDAWG. Defaults to 1.2e9.
        stepSize (float, optional): Target stepsize of time-based measurement in seconds. Defaults to 10e-9.
        Tmax (float, optional): Maximum time of experiment in seconds. Defaults to 5e-6.

    Returns:
       nPoints (int): Number of points in the waveform
       nSteps (int): Number of steps in the sequence
       pulse_length_increment (int): Number of points for stepsize
       pulse_length_start (int): Number of points for initial waveform
       """
    if sequence == 'rabi':
        base = 4
    else:
        base = 16
    pulse_length_start = int(roundToBase(stepSize*fsAWG,base=base))
    if pulse_length_start < 32 and sequence != 'rabi':
        print('Smallest Waveform Length is 32 samples. The first point in this sequence has %d samples'%(pulse_length_start))
        sys.exit()
    
    pulse_length_increment = roundToBase(fsAWG*stepSize,base=base)
    nPoints = roundToBase(Tmax*fsAWG,base=pulse_length_increment) # this ensures there is an integer number of time points
    nSteps = int((nPoints-pulse_length_start)/pulse_length_increment) + 1 # 1 is added to include the first point
   
    print("dt is %.1f ns (%d pts) ==> f_s = %.1f MHz \nNpoints = %d | n_steps is %d | Pulse length start = %.1f ns (%d pts)" %(pulse_length_increment/fsAWG*1e9,pulse_length_increment,1e-6*fsAWG/pulse_length_increment,nPoints,nSteps,pulse_length_start*1e9/fsAWG,pulse_length_start))
    if nSteps > 1024:
        print('Error: The maximum number of steps is 1024')
        sys.exit()
    return nPoints,nSteps,pulse_length_increment,pulse_length_start

def roundToBase(nPoints,base=16):
    '''Make the AWG happy by uploading a wfm whose points are multiple of 16'''
    y = base*round(nPoints/base)
    if y==0:
        y = base*round(nPoints/base+1)
    return y

# def pull_wfm(sweep_name,RT_pars,instance):
#     tel_amp = RT_pars[0]
#     tau = RT_pars[1]
#     nu = RT_pars[2]*1e6
#     path = "E:\\generalized-markovian-noise\\%s\\sweep_data\\ramsey\\%s\\noise_instances"%('CandleQubit_6',sweep_name)
#     # filename = 'nu_%d_kHz_tau_%d_ns.csv' %(round(nu*1e-3),round(tau*1e3))
#     filename = 'RTN_tau_%d_ns.csv' %(round(tau*1e3))
#     print(os.path.join(path,filename))
#     with open(os.path.join(path,filename)) as waveform_file:
#         noise_inst = np.array(next(itertools.islice(csv.reader(waveform_file),0,instance,1)),dtype=np.float32)
#         qubit_free_evol = tel_amp*noise_inst
#     return qubit_free_evol

def pull_wfm(sweep_name,RT_pars):
    """
    Pulls waveform from file for parameter sweep

    Args:
        sweep_name (str): Name of file where noise waveforms are stored
        RT_pars (list): Parameters of generalized Markovian noise

    Returns:
        noise_realizations(array): Array of noise waveforms
    """
    tel_amp = RT_pars[0]
    tau = RT_pars[1]
    nu = RT_pars[2]*1e6
    path = "E:\\generalized-markovian-noise\\%s\\sweep_data\\ramsey\\%s\\noise_instances"%('CandleQubit_6',sweep_name)
    filename = 'nu_%d_kHz_tau_%d_ns.csv' %(round(nu*1e-3),round(tau*1e3))
    # filename = 'RTN_tau_%d_ns.csv' %(round(tau*1e3))
    print(os.path.join(path,filename))
    noise_realizations = np.loadtxt(os.path.join(path,filename),dtype=float,delimiter=',')
    return noise_realizations


def create_wfm_file(AC_pars,RT_pars,nPoints,sweep,sweep_name,Tmax,instance,sequence="ramsey",meas_device='CandleQubit_6'):
    """
    Creates noise waveforms and saves them to file where they can be pulled from later by the HDAWG

    Args:
        AC_pars (list): List containing mean and standard deviation of white noise
        RT_pars (list): List containing amplitude, decay time, and frequency of RTN
        nPoints (int): Number of points in waveform
        sweep (bool): Whether it is a parmeter sweep or a single point in parameter space
        sweep_name (str): Name of sweep
        Tmax (float): Maximum length of waveform in seconds. 
        instance (int): Instance of noise waveform 
        sequence (str, optional): Specifies which sequence type is going to be used. Defaults to "ramsey".
        meas_device (str, optional): Name of measurement device. Defaults to 'CandleQubit_6'.
    """
    # create RTN noise or pull instance from file (only for parameter sweeps)
    if RT_pars[0]!= 0:
        tel_amp = RT_pars[0]
        tau = RT_pars[1]
        nu = RT_pars[2]*1e6
        if sweep == 0:
            t = np.linspace(0,Tmax,nPoints)
            qubit_free_evol = tel_amp * np.cos(2*np.pi*nu*t) * gen_tel_noise(nPoints, tau, dt=Tmax/nPoints)
        elif sweep == 1:
            path = "E:\\generalized-markovian-noise\\%s\\sweep_data\\ramsey\\%s\\noise_instances"%('CandleQubit_6',sweep_name)
            filename = "nu_%d_kHz_tau_%d_ns.csv" % (round(nu*1e-3),round(tau*1e3))
            print(os.path.join(path,filename))
            with open(os.path.join(path,filename)) as waveform_file:
                noise_inst = np.array(next(itertools.islice(csv.reader(waveform_file),instance,None)),dtype=np.float32)
                qubit_free_evol = tel_amp*noise_inst
    else:
        qubit_free_evol = np.zeros(nPoints)

    qubit_free_evol = qubit_free_evol[...,None]

    # create white noise instance
    if AC_pars[0] != 0 or AC_pars[1] != 0:
        white_noise = np.random.normal(loc=AC_pars[0], scale=AC_pars[1], size=nPoints)
        white_noise = white_noise[...,None]
        len(qubit_free_evol)
        len(white_noise)
        wfm_arr = np.hstack((qubit_free_evol,white_noise))
    else:
        wfm_arr = qubit_free_evol

    # save file
    if sequence == "ramsey":
        fileName = "ramsey_wfm"
        np.savetxt("C:/Users/LFL/Documents/Zurich Instruments/LabOne/WebServer/awg/waves/"+fileName+".csv", wfm_arr, delimiter = ",")
    elif sequence == "echo":
        fileName = "echo_wfm"
        np.savetxt("C:/Users/LFL/Documents/Zurich Instruments/LabOne/WebServer/awg/waves/"+fileName+".csv", wfm_arr, delimiter = ",")
    elif sequence == "T1":
        fileName = "T1_wfm"
        np.savetxt("C:/Users/LFL/Documents/Zurich Instruments/LabOne/WebServer/awg/waves/"+fileName+".csv", wfm_arr, delimiter = ",")

def gen_noise_realizations(par1_arr:array =np.linspace(0,10,100),par2_arr:list=[0],numRealizations:int=3,nPoints:int=1000,T_max:float=5e-6,sweep_count:int=1,meas_device:str='CandleQubit_6'):
    """
    Generates noise waveforms and saves them to file for use in parameter sweeps

    Args:
        par1_arr (array, optional): Array of values for first parameter of generalized markovian noise. Defaults to np.linspace(0,10,100).
        par2_arr (list, optional): Array of values for second parameter of generalized markovian noise. Defaults to [0].
        numRealizations (int, optional): Number of noise realizations for each parameter point. Defaults to 3.
        nPoints (int, optional): Number of points in noise waveform. Defaults to 1000.
        T_max (float, optional): Maximum duration of noise waveform in seconds. Defaults to 5e-6.
        sweep_count (int, optional): Index of sweep. Defaults to 1.
        meas_device (str, optional): Name of measurement device. Defaults to 'CandleQubit_6'.
    """
    
    numPoints_par1 = len(par1_arr)
    numPoints_par2 = len(par2_arr)
    t = np.linspace(0,T_max,nPoints)
    parent_dir = 'E:\\generalized-markovian-noise\\%s\\sweep_data\\ramsey\\'%(meas_device)
    directory = 'sweep_%03d\\noise_instances'%(sweep_count)
    path = os.path.join(parent_dir,directory)
    os.mkdir(path)
    noise_arr = np.zeros((numRealizations,nPoints))
    for i in range(numPoints_par2):
        for k in range(numPoints_par1):
            filename = "nu_%d_kHz_tau_%d_ns.csv" % (round(par2_arr[i]*1e3),round(par1_arr[k]*1e3))
            with open(os.path.join(path,filename),"w",newline="") as datafile:
                writer = csv.writer(datafile)
                for j in range(numRealizations):
                    if len(par2_arr) == 1 and par2_arr[0] == 0:
                        noise_arr[j,:] = gen_tel_noise(nPoints, par1_arr[k], dt = T_max/nPoints)
                    elif len(par2_arr) > 1:
                        noise_arr[j,:] = np.cos(2*np.pi*par2_arr[i]*t+2*np.pi*np.random.random(1)[0]) * gen_tel_noise(nPoints, par1_arr[k], dt = T_max/nPoints)

                writer.writerows(noise_arr)

def gen_tel_noise(numPoints,tau,dt):
    """
    Generates instance of telegraph noise

    Args:
        numPoints (int): Number of points in waveform
        tau (float): Decay time constant of noise
        dt (float): Step size of waveform

    Returns:
        [type]: [description]
    """
    signal = np.ones(numPoints)*(-1)**np.random.randint(0,2)
    for i in range(1,numPoints-1):
        if np.random.rand() < 1/(2*tau*1e-6/dt)*np.exp(-1/(2*tau*1e-6/dt)):
            signal[i+1] = - signal[i]
        else:
            signal[i+1] = signal[i]
    return signal

def odd(n):
    return range(1,n,2)

def even(n):
    return range(0,n,2)

def calc_timeout(nAverages,measPeriod,dt,nSteps):
    t = 0
    for i in range(nSteps):
        t += (dt*i+measPeriod)*nAverages
    return t

def init_arrays(numRealizations=128,interval=2,nPointsBackground=200,nPoints=200):
    bData_I = np.zeros((int(numRealizations/interval),nPointsBackground),dtype=float)
    bData_Q  = np.zeros((int(numRealizations/interval),nPointsBackground),dtype=float)

    data_I = np.zeros((numRealizations,nPoints),dtype=float)
    data_Q = np.zeros((numRealizations,nPoints),dtype=float)

    return bData_I,bData_Q,data_I,data_Q

def setup_active_reset(awg,daq,threshold=5):

    # if active
    # Configure AWG settings
    # select trigger sources
    daq.setInt('/dev8233/awgs/0/auxtriggers/0/channel', 0)
    daq.setInt('/dev8233/awgs/0/auxtriggers/1/channel', 1)
    # Select trigger slope. First trigger is QA Result TRigger (rise), second is QA Result (level)
    daq.setInt('/dev8233/awgs/0/auxtriggers/0/slope', 1)
    daq.setInt('/dev8233/awgs/0/auxtriggers/1/slope', 0)
    # sets trigger level
    daq.setDouble('/dev8233/triggers/in/0/level', 0.3)
    daq.setDouble('/dev8233/triggers/in/1/level', 0.3)
    #Configure QA settings
    # select trigger sources
    daq.setInt('/dev2528/triggers/out/0/source', 74)
    daq.setInt('/dev2528/triggers/out/1/source', 64)
    # set trigger mode to output ("drive")
    daq.setInt('/dev2528/triggers/out/0/drive', 1)
    daq.setInt('/dev2528/triggers/out/1/drive', 1)
    # set trigger levels to 3 V
    # daq.setDouble('/dev2528/triggers/in/0/level', 3)
    # daq.setDouble('/dev2528/triggers/in/1/level', 3)
    # sets QA result threshold
    daq.setDouble('/dev2528/qas/0/thresholds/0/level', threshold)

# def set_AWG_output_amplitude(range)

def calc_sweep_time(par1,par2,measTimeBackground=1,measTime=25,nMeasBackground=100,nMeas=100):
    return (measTimeBackground*nMeasBackground+measTime*nMeas)*len(par1)*len(par2)


