#!/usr/bin/env python
# coding: utf-8

# # Import Modules

import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import UHFQA as qa
import HDAWG as hd
#import USB6501 as usb
import zhinst.utils as ziut
from IPython.display import clear_output
import sys
import importlib.util
import smf100a as smf
# import my_math_funcsas  mmf
import zhinst as zi
import textwrap
import comTablefuncs as ctfuncs
import json
import csv
# import keyboard as kb
import os
import plot_functions as pf
import itertools
# from VISAdrivers.sa_api import *
import seaborn as sns; sns.set() # styling
import collections
from PyTektronixScope import PyTektronixScope

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def mixer_calib(awg,mixer='qubit',fc=3.875e9):
    """
    DESCRIPTION:
        Optimizes mixer at given frequency

    INPUTS:
        device (class): The instrument (AWG or QA) that controls the I and Q channels of the mixer we want to optimize
        mixer (string): The mixer we want to optimize. Options are "qubit","resonator",and "stark". Defaults to 'qubit'.
        f_c (float): The frequency we want to optimize at. Defaults to 3.875e9.
    """

    # Open device
    handle = sa_open_device()["handle"]
    # Configure device
    sa_config_center_span(handle, fc, 0.25e6)
    sa_config_level(handle, -70)
    sa_config_sweep_coupling(device = handle, rbw = 1e3, vbw = 1e3, reject=0)
    sa_config_acquisition(device = handle, detector = SA_AVERAGE, scale = SA_LOG_SCALE)
    sa_config_gain_atten(handle, SA_AUTO_ATTEN, SA_AUTO_GAIN, True)

    # Initialize
    sa_initiate(handle, SA_SWEEPING, 0)
    query = sa_query_sweep_info(handle)
    sweep_length = query["sweep_length"]
    start_freq = query["start_freq"]
    bin_size = query["bin_size"]

    freqs = [start_freq + i * bin_size for i in range(sweep_length)]

    if mixer == 'qubit':
        device = 'dev8233'
        channels = [0,3]
    elif mixer == 'ac':
        device = 'dev8233'
        channels = [1,2]
    elif mixer == 'readout':
        device = 'dev2528'
        channels = [0,1]
    # get initial offset values
    I_offset = awg.get('/%s/sigouts/%d/offset'%(device,channels[0]))['dev8233']['sigouts']['%d'%channels[0]]['offset']['value']
    Q_offset = awg.get('/%s/sigouts/%d/offset'%(device,channels[1]))['dev8233']['sigouts']['%d'%(channels[1])]['offset']['value']
    data = sa_get_sweep_64f(handle)['max']
    peak_P = max(data)
    peak_f = np.argmax(data)
    print("Calibrating mixer at %.4f GHz"%(freqs[peak_f]*1e-9))

    dV = 1e-3
    i = 0 # 0 when adjusting I offset for the first time, 1 otherwise
    q = 0 # 0 when adjusting Q offset for the first time, 1 otherwise
    j = 0 # keeps track of channel changes (I->Q or Q->I)
    k = 0 # determines which channel we are adjusting (0 is I, 1 is Q)

    # Do calibration
    while max(sa_get_sweep_64f(handle)['max']) > -90:
        # read offset from channel
        offset = awg.get('/%s/sigouts/%d/offset'%(device,k))['%s'%(device)]['sigouts']['%d'%k]['offset']['value']
        awg.setDouble('/%s/sigouts/%d/offset'%(device,k),offset+dV)
        awg.sync()
        dP = peak_P - max(sa_get_sweep_64f(handle)['max'])
        if dP > 0 and i == 0:
            # go in the other direction
            awg.setDouble('/%s/sigouts/%d/offset'%(device,k),I_offset-dV)
            awg.sync()
            i = 1
        elif dP > 0 or dP < abs(0.1) and i == 1:
            # switch to other channel
            if k == 0:
                k = 1
            elif k == 1:
                k = 0
            j += 1
        elif dP < 0:
            # keep going in the same direction until no decrease in OFF power
            offset = awg.get('/%s/sigouts/%d/offset'%(device,k))['%s'%(device)]['sigouts']['%d'%(k)]['offset']['value']
            awg.setDouble('%s/sigouts/%d/offset'%(device,k),offset-dV)
            awg.sync()
        elif j > 0:
            # decrease stepsize by a factor of 10
            dV = dV/10
        elif dV < 1e-5:
            break
        time.sleep(2)


    data = sa_get_sweep_64f(handle)['max']
    p_OFF = max(data)
    freq = np.argmax(data)
    print("Mixer Optimized\nOFF power is %.1f @ %.4f GHz"%(p_OFF,freq))

    # Device no longer needed, close it
    sa_close_device(handle)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(freqs,data)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Power (dBm)')

def get_power(freq=5.7991e9,threshold=-80,plot=0):

    '''
    DESCRIPTION: retrieves the power at the specified frequency
    '''
    # Open device
    saCloseDevice()
    handle = sa_open_device()["handle"]

    # Configure device
    sa_config_center_span(handle, freq, 0.1e6)
    sa_config_level(handle, threshold)
    sa_config_sweep_coupling(handle, 1e3, 1e3, 0)
    sa_config_acquisition(handle, SA_AVERAGE, SA_LOG_SCALE)
    sa_config_gain_atten(handle, SA_AUTO_ATTEN, SA_AUTO_GAIN, True)

    # Initialize
    sa_initiate(handle, SA_SWEEPING, 0)
    query = sa_query_sweep_info(handle)
    sweep_length = query["sweep_length"]
    start_freq = query["start_freq"]
    bin_size = query["bin_size"]
    freqs = [start_freq + i * bin_size for i in range(sweep_length)]

    # Get sweep
    sweep_max = sa_get_sweep_32f(handle)['max']

    # Device no longer needed, close it
    sa_close_device(handle)

    index_max = np.argmax(np.array(sweep_max))
    freq_max = freqs[index_max]
    p_max = sweep_max[index_max]

    if plot == 1:
        # Plot
        freqs = [start_freq + i * bin_size for i in range(sweep_length)]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(freqs,sweep_max)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Power (dBm)')

    print('Max Power is %.1f dBm at %.6f GHz' %(p_max,freq_max*1e-9))
    return freq_max,p_max


def readoutSetup(awg,device_awg,sequence='spec',readout_pulse_length=1.2e-6,cav_resp_time=0.5e-6):

    fs = 450e6
    readout_amp = 0.7
    # prepares QA for readout | loads readout Sequence into QA AWG and prepares QA
    print('-------------Setting up Readout Sequence-------------')
    if sequence =='spec':
        qa.awg_seq_readout(awg,device_awg, readout_length=readout_pulse_length,nPoints=roundToBase(readout_pulse_length*fs),base_rate=fs,cav_resp_time=cav_resp_time,amplitude_uhf=readout_amp)
        awg.setInt('/%s/awgs/0/time'%(device_awg),2)
    elif sequence =='pulse':
        qa.awg_seq_readout(awg,device_awg, readout_length=readout_pulse_length,nPoints=roundToBase(readout_pulse_length*fs),base_rate=fs,cav_resp_time=cav_resp_time,amplitude_uhf=readout_amp)
        awg.setInt('/%s/awgs/0/time'%(device_awg),2)

def pulsed_spec_setup(daq,device_qa,awg,device_awg,nAverages,qubit_drive_amp,AC_pars=[0,0],qubit_drive_dur=30e-6,result_length=1,integration_length=2e-6,nPointsPre=0,nPointsPost=0):

    hd.awg_seq(awg,device_awg,sequence='qubit spec',fs=0.6e9,nAverages=nAverages,qubit_drive_dur=qubit_drive_dur,AC_pars=AC_pars,amplitude_hd= qubit_drive_amp,nPointsPre=nPointsPre,nPointsPost=nPointsPost)
    awg.setInt('/%s/awgs/0/time'%(device_awg),2) # sets AWG sampling rate to 600 MHz
    qa.config_qa(daq, device_qa,sequence='spec',integration_length=integration_length,delay=0,nAverages=1,result_length=result_length)

def single_shot_setup(daq,device_qa,awg,device_awg,nAverages,qubit_drive_amp,AC_pars=[0,0],result_length=1,integration_length=2e-6):

    hd.awg_seq(awg,device_awg,sequence='single_shot',fs=0.6e9,nAverages=nAverages,AC_pars=AC_pars,amplitude_hd=qubit_drive_amp,measPeriod=10e-6)
    awg.setInt('/%s/awgs/0/time'%(device_awg),2) # sets AWG sampling rate to 600 MHz
    qa.config_qa(daq, device_qa,sequence='spec',integration_length=integration_length,delay=0,nAverages=1,result_length=result_length)


def seq_setup(awg,device_awg='dev8233',sequence='rabi',nAverages=128,nPoints=1024,pulse_length_start=32,fs=2.4e9,nSteps=100,pulse_length_increment=16,Tmax=0.3e-6,amplitude_hd=1,AC_pars=[0.4,0],pi2Width=0,piWidth_Y=0,pipulse_position=20e-9,measPeriod=200e-6,instance=0,sweep_name=0,sweep=0,RT_pars=[0,0]):
    """
    Function Description
    --------------------

    Sets up the right sequence to the AWG along with the corresponding command table

    Parameters
    ----------
    awg : class
        awg to write the sequence to
    device_awg : string
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
    if AC_pars[0] != 0:
        prePulseLength = 2500e-9 # AC stark tone is ON before and after the qubit drive tone
        postPulseLength = 1000e-9
        nPointsPre = roundToBase(prePulseLength*fs+pi2Width)
        nPointsPost = roundToBase(postPulseLength*fs+pi2Width)
    else:
        nPointsPre = 0
        nPointsPost = 0

    # Generate and compile program
    print('-------------Setting HDAWG sequence-------------')
    bt = time.time()
    awg.setInt('/dev8233/awgs/0/time',(int(fs_base/fs-1))) # set sampling rate of AWG to 2.4 GHz
    hd.awg_seq(awg,AC_pars=AC_pars,fs=fs,nPoints=nPoints,pulse_length_increment=pulse_length_increment,nSteps=nSteps,nPointsPre=nPointsPre,pi2Width=pi2Width,pipulse_position=round(fs*pipulse_position),piWidth_Y=piWidth_Y,nPointsPost=nPointsPost,Tmax=Tmax,amplitude_hd=amplitude_hd,nAverages=nAverages,sequence=sequence,measPeriod=measPeriod,RT_pars=RT_pars)
    et = time.time()
    print('HDAWG compilation duration: %.1f s'%(et-bt))

    # create and upload command table
    ct=ctfuncs.ct_pulse_length(n_wave=nSteps, pulse_length_start=pulse_length_start, pulse_length_increment=pulse_length_increment,AC_amp=AC_pars[0], AC_pars=[nPointsPre,nPointsPost,int(2*pi2Width)], sequence=sequence)
    awg.setVector("/dev8233/awgs/0/commandtable/data", json.dumps(ct))

    awg.sync()
    # print(flatten(ct))
    return nSteps,ct



def single_shot(daq,device_qa,awg,device_awg,qubitLO=0,cav_resp_time=1e-6,integration_length=2e-6,AC_pars=[0,0],qubit_drive_amp=1,readout_drive_amp=0.1,setup=0,nAverages=128):
    '''
    DESCRIPTION: Executes single shot experiment.

    '''
    result_length =  2*nAverages
    fsAWG = 600e6
    base_rate = 1.8e9

    readout_pulse_length = integration_length + cav_resp_time + 2e-6


    if not setup:
        single_shot_setup(daq, device_qa, awg, device_awg,result_length=result_length,AC_pars=AC_pars,integration_length=integration_length,nAverages=nAverages,qubit_drive_amp=qubit_drive_amp)
        readoutSetup(daq, device_qa, sequence='spec',readout_pulse_length=readout_pulse_length,cav_resp_time=cav_resp_time)
        time.sleep(0.1)

    # initialize signal generators and set power


    print('Start measurement')
    sweep_data, paths = qa.create_sweep_data_dict(daq, device_qa)
    data_pi = []
    data_OFF = []


    qa.enable_awg(daq, device_qa) # start the readout sequence
    bt = time.time()
    qa.qa_result_reset(daq, device_qa)
    qa.qa_result_enable(daq, device_qa)
    hd.enable_awg(awg,device_awg,awgs=[0]) #runs the drive sequence
    data = qa.acquisition_poll(daq, paths, result_length, timeout = 60) # transfers data from the QA result to the API for this frequency point
    # seperate OFF/ON data and average
    data_OFF = np.append(data_OFF, np.mean([data[paths[0]][k] for k in even(len(data[paths[0]]))]))
    data_pi =  np.append(data_pi, np.mean([data[paths[0]][k] for k in odd(len(data[paths[0]]))]))

    hd.enable_awg(awg, device_awg,enable=0,awgs=[0])
    qa.stop_result_unit(daq, device_qa, paths)
    qa.enable_awg(daq, device_qa, enable = 0)

# ----------------------------------------------------------------------------------
    et = time.time()
    duration = et-bt
    print(f'Measurement time: {duration} s')
#-----------------------------------

    return data_OFF,data_pi

def spectroscopy(daq,device_qa,awg,device_awg,qubitLO=0,cav_resp_time=1e-6,integration_length=2e-6,AC_pars=[0,0],qubit_drive_amp=1,readout_drive_amp=0.1,setup=0,nAverages=128,frequencies=np.linspace(3.7,3.95,1001)):
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
        pulsed_spec_setup(daq, device_qa, awg, device_awg,result_length=result_length,AC_pars=AC_pars,qubit_drive_dur=qubit_drive_dur,integration_length=integration_length,nAverages=nAverages,qubit_drive_amp=qubit_drive_amp,nPointsPre=nPointsPre,nPointsPost=nPointsPost)
        readoutSetup(daq, device_qa, sequence='spec',readout_pulse_length=readout_pulse_length,cav_resp_time=cav_resp_time)
        time.sleep(0.1)

    # initialize signal generators and set power


    print('Start measurement')
    sweep_data, paths = qa.create_sweep_data_dict(daq, device_qa)
    data_ON = []
    data_OFF = []


    qa.enable_awg(daq, device_qa) # start the readout sequence
    bt = time.time()
    j = 0
    for f in frequencies:
        qubitLO.set_freq(f)
        qa.qa_result_reset(daq, device_qa)
        qa.qa_result_enable(daq, device_qa)
        hd.enable_awg(awg,device_awg,awgs=[0]) #runs the drive sequence
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

    hd.enable_awg(awg, device_awg,enable=0,awgs=[0])
    qa.stop_result_unit(daq, device_qa, paths)
    qa.enable_awg(daq, device_qa, enable = 0)

# ----------------------------------------------------------------------------------
    et = time.time()
    duration = et-bt
    print(f'Measurement time: {duration} s')
#-----------------------------------

    return power_data,I_data,Q_data

def pulse(daq,awg,channel=0,setup=[0,0,0],Tmax=0.3e-6,nSteps=61,nAverages=128,amplitude_hd=1,sequence='rabi',AC_pars=[0,0],stepSize=2e-9,\
          RT_pars=[0,0],cav_resp_time=0.5e-6,piWidth_Y=0,AC_freq=5e-9,pipulse_position=20e-9,integration_length=1.2e-6,qubitDriveFreq=3.8135e9,run=1,pi2Width=0,sampling_rate=1.2e9,measPeriod=200e-6,sweep=0,sweep_name='sweep_001',instance=0):


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
    'pipulse_position':         Where to insert the pipulse (only applicable for echo with telegraph noise). A higher number means the pipulse is applied sooner
    cav_resp_time:              The time it takes for the cavity to ring up/down. Since we are using square integration pulse, we don't want to include the edges of the pulse
    integration_length:         How long the QA integrates for. The readout pulse is 2 microseconds longer than integration+cavity_response

    '''

    device_qa='dev2528'
    device_awg='dev8233'
    fs = sampling_rate
    base_rate = 1.8e9       # sampling rate of QA (cannot be chaned in standard mode)
    delay = 0# maximum delay is 1020 samples or 566.7 ns
    readout_pulse_length = integration_length + cav_resp_time + 2.0e-6

    if sequence == 'echo' and RT_pars[0] != 0:
        if int(fs*stepSize) < 64:
            print('Error: The minimum step size is 64 pts')
            sys.exit()
        cores = [1,0]
    else:
        cores = [0]

    hd.enable_awg(awg,device_awg,enable=0,awgs=cores)
    qa.enable_awg(daq, device_qa,enable=0)
    qa.qa_result_reset(daq, device_qa)

    if setup[0] == 0:
        nPoints,nSteps,pulse_length_increment,pulse_length_start = calc_nSteps(sequence=sequence,fsAWG=fs,piWidth_Y=piWidth_Y,stepSize=stepSize,Tmax=Tmax,RT_pars=RT_pars)
        if sequence == 'ramsey' or sequence == 'echo' or sequence == 'T1':
            create_wfm_file(AC_pars=AC_pars, RT_pars=RT_pars,sequence=sequence, nPoints=nPoints, sweep=sweep, sweep_name=sweep_name, Tmax=Tmax, instance=instance)
        nSteps,ct = seq_setup(awg,device_awg,sequence=sequence,piWidth_Y=piWidth_Y,pipulse_position=pipulse_position,nSteps=nSteps,nPoints=nPoints,fs=fs,pulse_length_start=pulse_length_start,pulse_length_increment=pulse_length_increment,instance=instance,sweep_name=sweep_name,amplitude_hd=amplitude_hd,nAverages=nAverages,pi2Width=pi2Width,Tmax=Tmax,AC_pars=AC_pars,RT_pars=RT_pars,measPeriod=measPeriod,sweep=sweep)
    elif setup[0] == 2:
        bt = time.time()
        # replace waveforms, don't recompile program
        noise_instance = pull_wfm(sweep_name=sweep_name, RT_pars=RT_pars, instance=instance)
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
        # if sequence == 'echo':
        #     nSteps = int(nSteps/2) + 1

    if setup[1] == 0:

        readoutSetup(daq,device_qa,readout_pulse_length=readout_pulse_length,sequence='pulse',cav_resp_time=cav_resp_time)

    if setup[2] == 0:
        qa.config_qa(daq, device_qa,sequence='pulse',nAverages=nAverages,integration_length=integration_length,result_length=nSteps,delay=delay)
        daq.sync()

    if AC_pars[0] != 0:
        use_ct = 1
    elif AC_pars[0] == 0 and sequence == 'rabi':
        use_ct = 1
    else:
        use_ct = 0
    print('Estimated Measurement Time: %d sec'%(calc_timeout(nAverages, measPeriod, stepSize, nSteps)))
    ct_awg = json.loads(daq.get(f"/{device_awg}/awgs/0/commandtable/data",flat=True)[f"/{device_awg}/awgs/0/commandtable/data"][0]['vector'])
    if ct_awg != ct and use_ct == 1:
        print('Error! Invalid Command Table used for Measurement\nCommand Table Sent to AWG\n\n%s\n\nCommand Table in AWG\n\n%s'%(ct,ct_awg))
        sys.exit()
    else:
        result_length = nSteps
        timeout = 2*nSteps*measPeriod*nAverages
        sweep_data, paths = qa.create_sweep_data_dict(daq, device_qa)

        qa.enable_awg(daq, device_qa,enable=1) # start the readout sequence

        qa.qa_result_enable(daq, device_qa)

        # print('----------------------------------\nStart %s measurement' %(sequence))
        # bt = time.time()
        str_meas = time.time()
        hd.enable_awg(awg,device_awg,enable=1,awgs=cores) #runs the drive sequence
        data = qa.acquisition_poll(daq, paths, num_samples = result_length, timeout = timeout)

        for path, samples in data.items():
            sweep_data[path] = np.append(sweep_data[path], samples)

        qa.stop_result_unit(daq, device_qa, paths)
        hd.enable_awg(awg, device_awg, enable = 0,awgs=cores)
        qa.enable_awg(daq, device_qa, enable = 0)
        end_meas = time.time()
        print('\nmeasurement duration: %.1f s' %(end_meas-str_meas))

        data = sweep_data[paths[0]][0:result_length]/(integration_length*base_rate)
        I = data.real
        Q = data.imag

        #Generate time array points

        t = np.zeros(nSteps)
        for i in range(nSteps):
            t[i] = ct['table'][i]['waveform']['length']/fs
        if sequence=='echo':
            t = 2*t

        return t,I,Q,nSteps
    # return I,Q,nSteps

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

def calc_nSteps(sequence='ramsey',fsAWG=1.2e9,stepSize=10e-9,nPoints=1024,Tmax=5e-6,RT_pars=[0,0],piWidth_Y=0):
    if sequence == 'rabi':
        base = 4
    else:
        base = 16
    pulse_length_start = int(roundToBase(stepSize*fsAWG,base=base))
    if pulse_length_start < 32 and sequence != 'rabi':
        print('Smallest Waveform Length is 32 samples. The first point in this sequence has %d samples'%(pulse_length_start))
        sys.exit()
    # if sequence == 'echo' and RT_pars[0] != 0:
    #     pulse_length_start = roundToBase(64+int(piWidth_Y*fsAWG))
    #     # if pulse_length_start % 16 != 0:
    #     #     print('Pulse length start point (%d) is not multiple of 16'%pulse_length_start)
    #     #     sys.exit()
    # elif sequence == 'rabi':
    #     pulse_length_start = 32
    # else:
    #     pulse_length_start = 32
    pulse_length_increment = roundToBase(fsAWG*stepSize,base=base)
    nPoints = roundToBase(Tmax*fsAWG,base=pulse_length_increment) # this ensures there is an integer number of time points
    nSteps = int((nPoints-pulse_length_start)/pulse_length_increment) + 1 # 1 is added to include the first point
    # if sequence == 'echo':
    #   pulse_length_increment = pulse_length_increment / 2
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

def pull_wfm(sweep_name,RT_pars,instance):
    tel_amp = RT_pars[0]
    tau = RT_pars[1]
    path = "E:\\generalized-markovian-noise\\noise_instances\\%s"%(sweep_name)
    filename = 'RTN_tau_%d_ns.csv' %(round(tau*1e3))
    print(os.path.join(path,filename))
    with open(os.path.join(path,filename)) as waveform_file:
        noise_inst = np.array(next(itertools.islice(csv.reader(waveform_file),instance,None)),dtype=np.float32)
        qubit_free_evol = tel_amp*noise_inst
    return qubit_free_evol

def create_echo_wfms(awg,fs=1.2e9,AC_pars=[0,0],RT_pars=[0,0],sweep_name='sweep_001',sweep=0,instance=0,nPoints=1024,Tmax=5e-6,pi2Width=50e-9,nSteps=101,pulse_length_increment=32):

    '''
    DESCRIPTION: Generates a series of waveforms to be uploaded into the AWG. The output is a series of csv files.
    '''

    start = time.time()
    ACpre = AC_pars[0]*np.ones(roundToBase(500e-9*fs))
    pi2 = np.ones(int(fs*pi2Width))
    pi2pre = 0 * ACpre
    ac_noise = np.random.normal(AC_pars[0], AC_pars[1], nPoints)
    if sweep == 0 and RT_pars[0] !=0:
        tel_noise = RT_pars[0]*gen_tel_noise(nPoints, RT_pars[1], dt=Tmax/nPoints)
    elif sweep == 1 and RT_pars[0] !=0:
        tel_noise = pull_wfm(sweep_name=sweep_name, RT_pars=RT_pars, instance=instance)
    else:
        tel_noise = np.zeros(nPoints)
    for i in range(nSteps):
        ch1_wfm = np.concatenate((pi2pre,pi2,tel_noise[0:i*pulse_length_increment],pi2,pi2,tel_noise[i*pulse_length_increment:2*i*pulse_length_increment],pi2,pi2pre))
        ch2_wfm = np.concatenate((ACpre,AC_pars[0]*pi2,ac_noise[0:i*pulse_length_increment],AC_pars[0]*pi2,AC_pars[0]*pi2,ac_noise[i*pulse_length_increment:2*i*pulse_length_increment],AC_pars[0]*pi2,ACpre))
        #pad with 0's at the end
        # if len(ch1_wfm)  % 16 != 0:
        #     ch1_wfm = np.pad(ch1_wfm,pad_width=(0,int(16-(len(ch1_wfm)%16))),mode='constant',constant_values=0)
        #     ch2_wfm = np.pad(ch2_wfm,pad_width=(0,int(16-(len(ch2_wfm)%16))),mode='constant',constant_values=0)
        ch1_wfm = ch1_wfm[...,None]
        ch2_wfm = ch2_wfm[...,None]
        wfm_2D_arr = np.hstack((ch1_wfm,ch2_wfm))
        np.savetxt("C:/Users/LFL/Documents/Zurich Instruments/LabOne/WebServer/awg/waves/"+"echo_"+"wfm_%03d"%(i)+".csv", wfm_2D_arr, delimiter = ",")

    end = time.time()
    print('Generating echo Waveforms took %.1f' %(end-start))

def create_wfm_file(AC_pars,RT_pars,nPoints,sweep,sweep_name,Tmax,instance,sequence="ramsey"):

    # create RTN noise or pull instance from file (only for parameter sweeps)
    if RT_pars[0]!=0:
        tel_amp = RT_pars[0]
        tau = RT_pars[1]
        if sweep == 0:
            qubit_free_evol = tel_amp * gen_tel_noise(nPoints, tau, dt=Tmax/nPoints)
        elif sweep == 1:
            path = "E:\\generalized-markovian-noise\\noise_instances\\%s"%(sweep_name)
            filename = 'RTN_tau_%d_ns.csv' %(round(tau*1e3))
            print(os.path.join(path,filename))
            with open(os.path.join(path,filename)) as waveform_file:
                noise_inst = np.array(next(itertools.islice(csv.reader(waveform_file),instance,None)),dtype=np.float32)
                qubit_free_evol = tel_amp*noise_inst
    else:
        qubit_free_evol = np.zeros(nPoints)

    qubit_free_evol = qubit_free_evol[...,None]

    # create white noise instance
    if AC_pars[0] != 0:
        white_noise = np.random.normal(loc=AC_pars[0], scale=AC_pars[1], size=nPoints)
        white_noise = white_noise[...,None]
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

def gen_noise_realizations(noiseType='RTN',par1_arr=np.linspace(0,10,100),par2_arr=[0],numRealizations=3,nPoints=1000,T_max=5e-6,sweep_count=1):

    numPoints_par1 = len(par1_arr)
    numPoints_par2 = len(par2_arr)
    t = np.linspace(0,T_max,nPoints)
    parent_dir = 'E:\\generalized-markovian-noise\\noise_instances\\'
    directory = 'sweep_%03d'%(sweep_count)
    path = os.path.join(parent_dir,directory)
    os.mkdir(path)
    noise_arr = np.zeros((numRealizations,nPoints))
    for i in range(numPoints_par2):
        for k in range(numPoints_par1):
            filename = "%s_tau_%d_ns.csv" % (noiseType,round(par1_arr[k]*1e3))
            with open(os.path.join(path,filename),"w",newline="") as datafile:
                writer = csv.writer(datafile)
                for j in range(numRealizations):
                    if noiseType == 'RTN':
                        noise_arr[j,:] = gen_tel_noise(nPoints, par1_arr[k], dt = T_max/nPoints)
                    elif noiseType == 'mod_RTN':
                        noise_arr[j,:] = np.cos(par2_arr[i]*t) * gen_tel_noise(nPoints, par1_arr[k], dt = T_max/nPoints)

                writer.writerows(noise_arr)

def gen_tel_noise(numPoints,tau,dt):

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

def qa_monitor_avg(daq,device_qa,length,averages):

    settings = [
        ("qas/0/monitor/enable", 0),
        ("qas/0/monitor/length", length),
        ("qas/0/monitor/averages", averages),
        ("qas/0/monitor/enable", 1),
        ("qas/0/monitor/reset", 1),
        ('qas/0/monitor/trigger/channel', 7)
    ]
    daq.set([(f"/{device_qa}/{node}", value) for node, value in settings])
    # Signals to measure

    paths = []
    for channel in range(2):
        path = f"/{device_qa:s}/qas/0/monitor/inputs/{channel:d}/wave"
        paths.append(path)
    daq.setInt(f'/{device_qa}/qas/0/monitor/reset', 1)
    daq.setInt(f'/{device_qa}/qas/0/monitor/enable', 1)

    daq.sync()
    time.sleep(0.2)
    # daq.setInt('dev2528/qas/0/monitor/trigger/channel',1)
    daq.subscribe(paths)

    # Perform acquisition
    print("Acquiring data...")
    data = qa.acquisition_poll(daq, paths, length,timeout=60)
    daq.setInt(f'/{device_qa}/qas/0/monitor/enable', 0)
    fig = plt.figure(figsize=(12,6))
    plt.plot(data[paths[0]])
    plt.plot(data[paths[1]])
    print(len(data[paths[0]]))
    print(len(data[paths[1]]))
    plt.title(f'Input signals after {averages:d} averages')
    plt.xlabel('nPoints')
    plt.ylabel('Amp (V)')
    plt.grid()
    plt.show()

    return data

def scope_meas(awg,daq,device_qa='dev2528',length=8192,nAverages=128,samp_rate=1.8e9,trigLevel=0.1):

    #setup and initialize scope
    scope = qa.config_scope(daq,device_qa,length,scope_avg_weight=1,scope_mode=0)
    scope = daq.scopeModule()
    daq.setInt('/dev2528/scopes/0/channel', 3)# 1: ch1; 2(DIG):ch2; 3: ch1 and ch2(DIG)
    daq.setInt('/dev2528/scopes/0/channels/0/inputselect', 0) # 0: sigin 1; 1: sigin 2
    daq.setDouble('/dev2528/scopes/0/length', length)
    scope.set('scopeModule/averager/weight', 1)
    scope.set('scopeModule/mode', 2)
    daq.setDouble('dev2528/scopes/0/length', length*nAverages)
    daq.setInt('/dev2528/scopes/0/single', 1)
    daq.setInt('/dev2528/scopes/0/trigchannel', 0)
    daq.setInt('/dev2528/scopes/0/trigenable', 1)
    daq.setDouble('/dev2528/scopes/0/trigholdoff', 50e-6) #In units of second. Defines the time before the trigger is rearmed after a recording event
    daq.setDouble('/dev2528/scopes/0/triglevel', trigLevel) #in Volts
    # daq.setInt('/dev2528/scopes/0/segments/enable', 1)
    daq.setInt('/dev2528/scopes/0/time', int(np.log2(1.8e9/samp_rate))) #set sampling rate
    daq.setInt('/dev2528/scopes/0/channel', 3) # enables both signal inputs
    # daq.setInt('/dev2528/scopes/0/segments/count',nAverages)
    daq.setDouble('/dev2528/sigins/0/range',0.4)
    daq.sync()

    qa.restart_avg_scope(scope)
    qa.enable_scope(daq,device_qa,enable=1)
    qa.subscrib_scope(scope,device_qa)
    qa.execute_scope(scope)

    hd.enable_awg(awg,'dev8233',enable=1,awgs=[0])

    while int(scope.progress()) != 1:
        time.sleep(0.05)
        result = scope.read()

    ch1Data = result['%s' % device_qa]['scopes']['0']['wave'][0][0]['wave'][0]/2**15
    ch2Data = result['%s' % device_qa]['scopes']['0']['wave'][0][0]['wave'][1]/2**15
    avgCh1Data = np.zeros(length)
    avgCh2Data = np.zeros(length)

    # for k in range(nAverages):
    #     avgCh1Data = avgCh1Data + ch1Data[k:k+length]
    #     avgCh2Data = avgCh2Data + ch2Data[k:k+length]

    # lines = plt.plot(avgCh1Data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ch1Data)
    ax.plot(ch2Data)

    qa.enable_scope(daq,device_qa,enable=0)
#    scope.set('scopeModule/clearhistory', 0)
    qa.finish_scope(scope)

    # return scope,avgCh1Data,avgCh2Data
    return scope,ch1Data,ch2Data

def calc_timeout(nAverages,measPeriod,dt,nSteps):
    t = 0
    for i in range(nSteps):
        t += (dt*i+measPeriod)*nAverages
    return t