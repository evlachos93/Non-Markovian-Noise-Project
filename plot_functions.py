# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:13:39 2021

@author: lfl
"""
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import scipy as sp
import scipy as scy
from matplotlib import cm
import sympy as sy
import csv
import itertools
from scipy.interpolate import interp1d
import scipy.fftpack
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from types import SimpleNamespace
pi=np.pi
import zhinst.utils as ziut
import seaborn as sns; sns.set() # styling
sns.set_style('ticks')

def spec_plot(freq,I,Q,readout_power=-30,qubit_drive_amp=0.2):

    freq = freq*1e9
    I = I*1e3
    mag = np.abs(I*I.conjugate()+Q*Q.conjugate())

    phase = np.unwrap(np.angle(I+1j*Q))
    sigma = np.std(I)
    peak = scy.signal.find_peaks(I,height=np.mean(I)+sigma,distance=100)
    # print(peak[0])
    peaks = peak[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(freq,phase,'-o', markersize = 3, c='C0')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Phase (rad)')
    textstr = "$P_r = %.1f$ dBm\n Qubit Wfm Amp = %.1f mV" %(readout_power,qubit_drive_amp*1e3)
    plt.gcf().text(1, 0.25, textstr, fontsize=14)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(freq,I,'-o', markersize = 3, c='C0')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('$I_{ON}-I_{OFF}$ (mV)')
    textstr = "$P_r = %.1f$ dBm\n Qubit Wfm Amp = %.1f mV" %(readout_power,qubit_drive_amp*1e3)
    plt.gcf().text(1, 0.25, textstr, fontsize=14)

    txt = "\n"
    for i in peaks:
        txt += "%.4f GHz \n" %(freq[i]*1e-9)
    print('Peaks are at: %s'%txt)

def fit_data(x_vector,y_vector,sequence='rabi',dt=0.01,qubitDriveFreq=3.8e9,amplitude_hd=1,AC_pars=[0,0],RT_pars=[0,0],
                 pi2Width=50e-9,piWidth_Y=0,fitting = 1, plot=1,save_fig = 0,iteration=1,nAverages=1,sampling_rate=2.4e9,
                 integration_length=2e-6,AC_freq=5e9,cav_resp_time=5e-6,sweep=0,stepSize=5e-6,Tmax=5e-6,measPeriod=5e-6,nSteps=101,
                 prePulseLength=1500e-9,postPulseLength=1500e-9,threshold=10e-3,active_reset=False):

    '''
    fit experiment data

    sequence:          'Rabi', 'T1' or 'T2'
    complex_amplitude:  complex amplitude from the measurement
    x_vector:           x data
    fitting:     0:     do not fit
                  1:     do fit
    save_fig:    0:     do not save
                  1:     do save
    '''
    x_vector = x_vector*1e6
    abs_camplitude = y_vector*1e3

    if sequence == "rabi":
        amp = (max(abs_camplitude)-min(abs_camplitude))/2
        offset = np.mean(abs_camplitude)
        period = 1e3/(extract_freq(x_vector*1e3, abs_camplitude, dt,plot=0))
        # period = 110
        print('Period Initial Guess: %.1f ns'%(period))
        phase = 0
        p0 = [amp,period,phase,offset]
        fitted_pars, covar = scy.optimize.curve_fit(rabi, x_vector*1e3, abs_camplitude,p0=p0,xtol=1e-6,maxfev=3000)
        pi_pulse = np.round(1/2*fitted_pars[1])
        error = np.sqrt(abs(np.diag(covar)))
        print("Pi pulse duration is %.1f ns"%(pi_pulse))

        # return pi_pulse,error

    elif sequence == "ramsey":
        amp = abs_camplitude[0]-abs_camplitude[-1]
        offset = np.mean(abs_camplitude)
        f = extract_freq(x_vector, abs_camplitude,dt)
        print('Initial Guess for Freq:%.4f MHz'%(f))
        if x_vector[-1] > 10:
            tau = 10
        else:
            tau = 0.2
        phase = 0
        p0 = [amp,f,phase,tau,offset]
        try:
            fitted_pars, covar = scy.optimize.curve_fit(ramsey, x_vector, abs_camplitude,p0=p0,xtol=1e-6,maxfev=6000)
            detuning = fitted_pars[1]
            T_phi = fitted_pars[3]
            error = np.sqrt(abs(np.diag(covar)))
        except:
            print('fitting failed')
            fitted_pars = np.zeros(5)
            detuning = 0
            T_phi = 0
            error = 20*np.ones(5)

        # return detuning,T_phi,error

    elif sequence == "echo":
        amp = abs_camplitude[0]-abs_camplitude[-1]
        offset = np.mean(abs_camplitude)
        if x_vector[-1] < 2:
            tau = 0.6
        else:
            tau = 1.5
        p0 = [amp,tau,offset]
        try:
            fitted_pars, covar = scy.optimize.curve_fit(decay, x_vector, abs_camplitude,p0=p0,xtol=1e-6,maxfev=6000)
            T_2 = fitted_pars[1]
            error = np.sqrt(abs(np.diag(covar)))
            # if sum(error) > 2: # try fitting again, this time excluding the first few points
            #     fitted_pars, covar = scy.optimize.curve_fit(decay, x_vector[10:], abs_camplitude[10:],p0=p0,xtol=1e-6,maxfev=6000)
            #     T_2 = fitted_pars[1]
            #     error = np.sqrt(abs(np.diag(covar)))
        except:
            print('fitting failed')
            fitted_pars = np.zeros(3)
            T_2 = 0
            error = 20*np.ones(3)

        # return T_2,error

    elif sequence == "T1":
        amp = abs_camplitude[0]-abs_camplitude[-1]
        offset = np.mean(abs_camplitude)
        tau = 2
        p0 = [amp,tau,offset]
        try:
            fitted_pars, covar = scy.optimize.curve_fit(decay, x_vector, abs_camplitude,p0=p0,xtol=1e-6,maxfev=3000)
            T_1 = fitted_pars[1]
            error = np.sqrt(abs(np.diag(covar)))
        except:
            print('fitting failed')
            fitted_pars = np.zeros(3)
            T_1 = 0
            error = 20*np.ones(3)

    return fitted_pars,error

def plot_data(awg,x_vector,y_vector,sequence='rabi',dt=0.01,qubitDriveFreq=3.8e9,amplitude_hd=1,AC_pars=[0,0],RT_pars=[0,0],
                 pi2Width=50e-9,piWidth_Y=0,fitting = 1, plot=1,save_fig = 0,iteration=1,nAverages=1,sampling_rate=2.4e9,
                 integration_length=2e-6,AC_freq=5e9,cav_resp_time=5e-6,sweep=0,stepSize=5e-6,Tmax=5e-6,measPeriod=5e-6,nSteps=101,
                 prePulseLength=1500e-9,postPulseLength=1500e-9,threshold=10e-3,active_reset=False,fitted_pars=[],pi_pulse=50,
                 fit_single_par_point=0):

    if fit_single_par_point == 0:
        x_vector = x_vector*1e6
        y_vector = y_vector*1e3
    elif fit_single_par_point == 1:
        x_vector[0] = x_vector[0]*1e6
        y_vector[0] = y_vector[0]*1e3
        x_vector[1] = x_vector[1]*1e6
        y_vector[1] = y_vector[1]*1e3

    if sequence == "rabi":
        fig, ax = plt.subplots()
        ax.plot(x_vector*1e3, y_vector, '-o', markersize = 3, c='C0')
        ax.set_ylabel('Digitizer Voltage (mV)')
        ax.set_xlabel('Pulse Duration (ns)')
        ax.plot(x_vector*1e3,rabi(x_vector*1e3, fitted_pars[0], fitted_pars[1], fitted_pars[2],fitted_pars[3]),'r')
        ax.set_title('Rabi Measurement %03d'%(iteration))
        textstr = '$\omega_d$ = %.4f GHz\n$A_d$ = %.2f V\n$T_{\pi/2}$ = %.1f ns\n$\mu$ = %d mV\n$\omega_{AC}$ = %.4f GHz\n$\hatn$ = %d'%(qubitDriveFreq*1e-9,amplitude_hd,round(fitted_pars[1]/4,1),AC_pars[0]*1e3,AC_freq*1e-9,nAverages)
        plt.gcf().text(0.95, 0.15, textstr, fontsize=14)

    elif sequence == "ramsey":
        if RT_pars[0] != 0 or AC_pars[0] != 0: # if noise is injected, sz or sx, plot data and noise instances
            fig = plt.figure(figsize=(20,18))
            ax1 = fig.add_subplot(2,2,(1,2))
            # Plot data
            fontSize = 36
            tickSize = 24
            markersize= 12
            linewidth = 6
            if sweep == 1 and fit_single_par_point == 1: # for plotting data averaged over many noise realizations post par sweep
                ax1.plot(x_vector[0], y_vector[0], '-o', markersize = markersize, c='C0',label='$B_0 = 0$')
                ax1.plot(x_vector[1], y_vector[1], '-o', markersize = markersize, c='k', label='$B_0 = %.2f mV$ | $\\tau_k = %.3f \mu s$'%(RT_pars[0]*1e3,RT_pars[1]))
                ax1.set_ylabel('Digitizer Voltage (mV)',fontsize=fontSize)
                ax1.set_xlabel('Pulse Separation ($\mu$s)')
                for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
                    label.set_fontsize(tickSize)
                ax1.plot(x_vector[0],ramsey(x_vector[0], fitted_pars[0][0], fitted_pars[0][1], fitted_pars[0][2],fitted_pars[0][3],fitted_pars[0][4]),'r',linewidth=linewidth)
                ax1.plot(x_vector[1],ramsey(x_vector[1], fitted_pars[1][0], fitted_pars[1][1], fitted_pars[1][2],fitted_pars[1][3],fitted_pars[1][4]),'g',linewidth=linewidth)
                ax1.legend(loc='upper right')
                # textstr = '$T_{\pi/2}$=%.1f ns\n$\omega_d$ = %.5f GHz\n$A_d$ = %.3f V\n$\Delta$=%.3f MHz\n$\\tau$/$\\tau_0$=%.2f\n$\mu$ = %d mV\n$\omega_{AC}$ = %.4f GHz\n$\sigma$ = %.1f mV\n$\hatn$ = %d'%(pi2Width*1e9,qubitDriveFreq*1e-9,amplitude_hd,fitted_pars[0][1],fitted_pars[1][3]/fitted_pars[0][3],float(AC_pars[1:AC_pars.find(' ')])*1e3,AC_freq*1e-9,float(AC_pars[AC_pars.find(' ')+1:])*1e3,nAverages)
                textstr = '$T_{\pi/2}$=%.1f ns\n$\omega_d$ = %.5f GHz\n$A_d$ = %.3f V\n$\Delta$=%.3f MHz\n$\\tau$/$\\tau_0$=%.2f\n$\mu$ = %d mV\n$\omega_{AC}$ = %.4f GHz\n$\sigma$ = %.2f mV\n$\hatn$ = %d'%(pi2Width*1e9,qubitDriveFreq*1e-9,amplitude_hd,fitted_pars[0][1],fitted_pars[1][3]/fitted_pars[0][3],AC_pars[0]*1e3,AC_freq*1e-9,AC_pars[1]*1e3,nAverages)
                plt.gcf().text(0.95, 0.15, textstr, fontsize=fontSize)
            elif (sweep == 0 or sweep == 1) and fit_single_par_point == 0: #for plotting non-par sweep data and par sweep data during the sweep
                ax1.plot(x_vector, y_vector, '-o', markersize = markersize, c='C0')
                ax1.set_ylabel('Digitizer Voltage (mV)',fontsize=fontSize)
                for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
                    label.set_fontsize(tickSize)
                ax1.plot(x_vector,ramsey(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2],fitted_pars[3],fitted_pars[4]),'r',linewidth=linewidth)
                # Retrieve and plot telegraph noise instrance
                waveforms = ziut.parse_awg_waveform(awg.get('/dev8233/awgs/0/waveform/waves/0')['dev8233']['awgs']['0']['waveform']['waves']['0'][0]['vector'],channels=2)
                t_arr = np.linspace(0,x_vector[-1],len(waveforms[0]))
                ax2 = fig.add_subplot(2,2,3)
                ax2.plot(t_arr,1e3*waveforms[0],linewidth=linewidth,color='b')
                ax3 = fig.add_subplot(2,2,4)
                ax3.plot(t_arr,1e3*waveforms[1],linewidth=linewidth,color='r')
                for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
                    label.set_fontsize(tickSize)
                for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
                    label.set_fontsize(tickSize)
                ax2.set_xlabel('Pulse Separation ($\mu$s)',fontsize=fontSize)
                ax2.set_ylabel('Amplitude (mV)',fontsize=fontSize)
                ax2.set_title('$\sigma_x$ waveform',fontsize=fontSize)
                ax3.set_title('$\sigma_z$ waveform',fontsize=fontSize)
                textstr = '$T_{\pi/2}$=%.1f ns\n$\omega_d$ = %.6f GHz\n$A_d$ = %.3f V\n$\Delta$=%.3f MHz\n$T_2^*$=%.2f $\mu$s\n$\mu$ = %d mV\n$\omega_{AC}$ = %.4f GHz\n$\sigma$ = %.1f mV\n$B_0$ = %.2f mV\n$\\tau_k$ = %.3f $\mu s$\n$\hatn$ = %d'%(pi2Width*1e9,qubitDriveFreq*1e-9,amplitude_hd,fitted_pars[1],fitted_pars[3],AC_pars[0]*1e3,AC_freq*1e-9,AC_pars[1]*1e3,RT_pars[0]*1e3,RT_pars[1],nAverages)
                plt.gcf().text(0.95, 0.25, textstr, fontsize=fontSize)
        else: # plot data without any noise instances
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            fontSize = 16
            tickSize = 11
            markersize= 6
            linewidth = 2
            ax1.plot(x_vector, y_vector, '-o', markersize = markersize, c='C0')
            ax1.set_ylabel('Digitizer Voltage (mV)',fontsize=fontSize)
            ax1.set_xlabel('Pulse Separation ($\mu$s)')
            for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
                label.set_fontsize(tickSize)
            ax1.plot(x_vector,ramsey(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2],fitted_pars[3],fitted_pars[4]),'r',linewidth=linewidth)
            textstr = '$T_{\pi/2}$=%.1f ns\n$\omega_d$ = %.6f GHz\n$A_d$ = %.3f V\n$\Delta$=%.3f MHz\n$T_2^*$=%.2f $\mu$s\n$\mu$ = %d mV\n$\omega_{AC}$ = %.4f GHz\n$\sigma$ = %.1f mV\n$B_0$ = %.1f mV\n$\\tau_k$ = %.3f $\mu s$\n$\hatn$ = %d'%(pi2Width*1e9,qubitDriveFreq*1e-9,amplitude_hd,fitted_pars[1],fitted_pars[3],AC_pars[0]*1e3,AC_freq*1e-9,AC_pars[1]*1e3,RT_pars[0]*1e3,RT_pars[1],nAverages)
            plt.gcf().text(0.95, 0.15, textstr, fontsize=fontSize)

        ax1.set_title('Ramsey Measurement %03d' %(iteration),size=fontSize)

    elif sequence == "echo":
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_vector, y_vector, '-o', markersize = 3, c='C0')
        ax.set_ylabel('Digitizer Voltage (mV)')
        ax.set_xlabel('Pulse Separation ($\mu$s)')
        ax.plot(x_vector,decay(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2]),'r')
        textstr = '$T_{\pi/2}$=%.1f ns\n$T_{\pi(Y)} = %.1f$ ns\n$\omega_d$ = %.4f GHz\n$A_d$ = %.2f V\n$T_2$=%.2f $\mu$s\n$\mu$ = %.3f V\n$\omega_{AC}$ = %.4f GHz\n$\sigma$ = %.3f V\n$B_0$ = %.3f V\n$\\tau_k$ = %.2f $\mu s$\n$\hatn$ = %d'%(pi2Width*1e9,piWidth_Y*1e9,qubitDriveFreq*1e-9,amplitude_hd,fitted_pars[1],AC_pars[0],AC_freq*1e-9,AC_pars[1],RT_pars[0],RT_pars[1],nAverages)
        ax.set_title('Echo Measurement %03d' %(iteration))
        plt.gcf().text(0.95, 0.15, textstr, fontsize=14)

    elif sequence == "T1":
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_vector, y_vector, '-o', markersize = 3, c='C0')
        ax.set_ylabel('Digitizer Voltage (mV)')
        ax.set_xlabel('Delay ($\mu$s)')
        ax.plot(x_vector,decay(x_vector, fitted_pars[0], fitted_pars[1], fitted_pars[2]),'r')
        textstr = '$T_{\pi/2}$=%.1f ns\n$\omega_d$ = %.4f GHz\n$A_d$ = %.2f V\n$T_1$=%.2f $\mu$s\n$\mu$ = %.3f V\n$\omega_{AC}$ = %.4f GHz\n$\sigma$ = %.3f V\n$B_0$ = %.2f V\n$\\tau_k$ = %.2f $\mu s$\n$\hatn$ = %d'%(pi2Width*1e9,qubitDriveFreq*1e-9,amplitude_hd,fitted_pars[1],AC_pars[0],AC_freq*1e-9,AC_pars[1],RT_pars[0],RT_pars[1],nAverages)
        ax.set_title('T1 Measurement %03d' %(iteration))
        plt.gcf().text(0.95, 0.15, textstr, fontsize=14)


def fit_beats(sequence,x_vector,y_vector,dt=0.01,qubitDriveFreq=3.8e9,amplitude_hd=1,AC_pars=[0,0],RT_pars=[0,0],pi2Width=50e-9,fitting = 1, plot=1,save_fig = 0,iteration=1):
    x_vector = x_vector*1e6
    abs_camplitude = np.abs(y_vector*1e3)
    amp = abs_camplitude[0]-abs_camplitude[-1]
    offset = np.mean(abs_camplitude)
    f1 = 15
    f2 = 1
    tau = 15
    phi1 = 0
    phi2 = 0
    p0 = [amp,f1,f2,phi1,phi2,tau,offset]
    lb = [-1000,-10,-10,-np.pi,-np.pi,0,-np.inf]
    ub = [1000,20,20,np.pi,np.pi,30,np.inf]
    fitted_pars, covar = scy.optimize.curve_fit(beats, x_vector, abs_camplitude,p0=p0,bounds=(lb,ub),xtol=1e-12,maxfev=6000)
    f1 = best_vals[1]
    f2 = best_vals[2]
    T_phi = best_vals[5]
    error = np.sqrt(abs(np.diag(covar)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_vector, abs_camplitude, '-o', markersize = 3, c='C0')
    ax.set_ylabel('Digitizer Voltage (mV)')
    ax.set_xlabel('Pulse Separation ($\mu$s)')
    ax.plot(x_vector,beats(x_vector,best_vals[0],best_vals[1],best_vals[2],best_vals[3],best_vals[4],best_vals[5],best_vals[6]),'r')
    textstr = '$A$ = %.2f mV\n$\omega_1$=%.2f MHz\n$\omega_2$=%.2f MHz\n$T_2^*$=%.2f $\mu$s\n$B_0$ = %.2f V\n$\\tau_k$ = %.2f $\mu s$'%(best_vals[0],best_vals[1],best_vals[2],best_vals[5],RT_pars[0],RT_pars[1])
    ax.set_title('Ramsey Measurement %03d' %(iteration))
    plt.gcf().text(1, 0.25, textstr, fontsize=14)

def sweep_plot_tau(x_data,y_data,z_data):
    fig = plt.figure(figsize=(3,3),constrained_layout=True)
    ax = fig.add_subplot(111,projection="3d")
    X,Y = np.meshgrid(x_data,y_data)
    surf = ax.plot_surface(X,Y*1e3,z_data, cmap = cm.coolwarm)

    SMALL_SIZE = 14
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 20
    # ax.set_zlim(0,2)
    ax.set_xlabel("$\\tau_k (\mu s)$")
    ax.set_ylabel("$B_0 (mV)$")
    ax.set_zlabel("$\\tau/\\tau_0 $")
    ax.view_init(elev=30,azim=-120)

    plt.xticks(rotation=60)
    plt.yticks(rotation=60)
    ax.set_xlim((0,max(x_data)))
    # cbr = fig.colorbar(surf,shrink=0.5)
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    # ax.set_yticklabels(ax.get_yticks(), rotation = 45)
    # ax.set_zlabel(ax.get_zlabel(), rotation = 45)
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # ax.tick_params(axis='both', which='major', pad=2)
    # plt.tight_layout(pad=1, w_pad=1, h_pad=1.0)
    plt.show()

def sweep_plot_omega(x_data,y_data,z_data):
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

    X,Y = np.meshgrid(x_data,y_data)
    surf = ax.plot_surface(X,Y,z_data, cmap = cm.coolwarm)

    ax.set_zlim(0,7)
    ax.set_xlabel("$\\tau_k (\mu s)$",loc='left')
    ax.set_ylabel("$B_0 (mV)$")
    ax.set_zlabel("$\omega (MHz)$")
    fig.colorbar(surf,shrink=0.5)

    plt.show()


def fit_sweep(par1_arr,par2_arr,sweep='sweep_001',plot=1,Tmax=5e-6,Tmax_b=10e-6,sequence='echo'):

    T2_b_arr = np.zeros((len(par1_arr),len(par2_arr)))
    detun_b_arr = np.zeros((len(par1_arr),len(par2_arr)))
    T2_arr = np.zeros((len(par1_arr),len(par2_arr)))
    detun_arr = np.zeros((len(par1_arr),len(par2_arr)))

    for i in range(len(par2_arr)):
        for j in range(len(par1_arr)):
            # start = time.time()
            t_b, data_b , t , data = extract_data(sweep,par1=par1_arr[j], par2 = par2_arr[i])
            print('Now fitting B0 = %.3f and tau = %.3f' %(par1_arr[j],par2_arr[i]))
            T2_arr[j,i], error = pulse_plot1d(sequence=sequence, x_vector=t, y_vector=np.mean(data,axis=0),dt=Tmax*1e6/data.shape[1], plot=plot,AC_pars=[0.6,0.08],RT_pars=[par1_arr[j],par2_arr[i]])
            # fit_beats(sequence='ramsey', dt=Tmax/data.shape[1], plot=plot,x_vector=t, y_vector=-np.mean(data,axis=0),AC_pars=[0.6,0.08],RT_pars=[par1_arr[j],par2_arr[i]])
            T2_b_arr[j,i], error = pulse_plot1d(sequence=sequence, x_vector=t_b, y_vector=np.mean(data_b,axis=0),dt=Tmax_b*1e6/data_b.shape[1],plot=plot, AC_pars=[0.6,0])
            end = time.time()
            # print('fitting one point took:%.2f sec'%(end-start))

    return detun_arr, T2_arr, detun_b_arr, T2_b_arr

def fit_single_instance(par1,par2,sweep='sweep_001',plot=1):
    # start = time.time()
    t_b, data_b , t , data = extract_data(sweep,par1=par1, par2 = par2)
    # data = data[instance,:]
    # detun , T2, error = pulse_plot1d(sequence='ramsey', dt=20/data.shape[1], plot=plot,x_vector=t, y_vector=np.mean(data,axis=0),AC_pars=[0.6,0.08],RT_pars=[par1,par2])
    fit_beats(sequence='ramsey', dt=5/data.shape[1], plot=plot,x_vector=t, y_vector=-np.mean(data,axis=0),AC_pars=[0.6,0.08],RT_pars=[par1,par2])
    end = time.time()

    # return detun, T2

def extract_data(sweep,par1,par2,meas_device):

    filename = 'RTN_B0_%d_mV_tau_%d_ns' %(round(par1*1e3),round(par2*1e3))
    with open("E:\\generalized-markovian-noise\\%s\\sweep_data\\ramsey\\%s\\data\\data_%s.csv"%(meas_device,sweep,filename)) as datafile:
        csv_reader = csv.reader(datafile,delimiter=',')
        file_data = list(csv_reader)

        for row in file_data:
            if row[0] == "Background Time Data":
                tdata_background = np.array(file_data[file_data.index(row)+1],dtype=float)
            if row[0] == "Time Data":
                tdata= np.array(file_data[file_data.index(row)+1],dtype=float)
            if row[0] == "Background Data: Channel 1":
                line_start_background = file_data.index(row) + 1
            if row[0] == "Background Data: Channel 2":
                line_end_background = file_data.index(row) - 1
            if row[0] == "Data: Channel 1":
                line_start = file_data.index(row) + 1
            if row[0] == "Data: Channel 2":
                line_end = file_data.index(row) - 1


        datafile.seek(0)
        # extract traces
        ydata_background = np.zeros((line_end_background-line_start_background+1,len(tdata_background)))
        line = line_start_background
        while line <= line_end_background:
            datafile.seek(0)
            ydata_background[line-line_start_background,:] = np.array(next(itertools.islice(csv_reader, line,None)),dtype=np.float32)
            line += 1

        datafile.seek(0)
        trace = np.array(next(itertools.islice(csv_reader, line_start,None)),dtype=np.float32)
        # ydata = np.zeros((line_end-line_start+1,len(trace)))
        ydata = np.zeros((line_end-line_start+1,len(tdata)))
        line = line_start
        while line <= line_end:
            datafile.seek(0)
            # data = np.array(next(itertools.islice(csv_reader, line,None)),dtype=np.float32)
            # ydata[line-line_start,:] = data[:148]
            ydata[line-line_start,:] = np.array(next(itertools.islice(csv_reader, line,None)),dtype=np.float32)[:]
            line += 1

        #extract data point parameters (qubitDriveFreq, ACstarkFreq, etc.)
        datafile.seek(0)
        keys = file_data[0]
        values =  file_data[1]
        dictionary = dict.fromkeys(keys,0)

        i = 0
        for key in dictionary:
            if key == 'AC_pars' or key == 'RT_pars':
                pars = np.zeros((2,1))
                values[i] = values[i].replace('[','')
                values[i] = values[i].replace(']','')
                values[i] = values[i].replace(',','')
                pars[0] = float(values[i][1:values[i].find(' ')])
                pars[1] = float(values[i][values[i].find(' ')+1:-1])
                dictionary[key] = pars
            try:
                dictionary[key] = float(values[i])
            except:
                dictionary[key] = values[i]
            i += 1

    return tdata_background, ydata_background, tdata, ydata, dictionary

def plot_single_par_point(par1,par2,sweep,fit=0,meas_device='CandleQubit_6'):
    '''
    DESCRIPTION: Plots the averaged ramsey trace over the different noise realizations
    '''
    tdata_background,ydata_background, tdata, ydata, exp_pars = extract_data(sweep=sweep, par1=par1, par2=par2,meas_device=meas_device)
    # average
    ydata_avg_background = np.mean(ydata_background,axis=0)
    ydata_avg = np.mean(ydata,axis=0)
    #fit
    fitted_pars,error = fit_data(x_vector=tdata, y_vector=ydata_avg,dt=tdata[-1]/len(tdata),**exp_pars)
    fitted_pars_background,error_b = fit_data(x_vector=tdata_background, y_vector=ydata_avg_background,dt=tdata_background[-1]/len(tdata_background),**exp_pars,fit_single_par_point=1)
    #plot
    plot_data(awg=None,x_vector=[tdata_background,tdata],y_vector=[ydata_avg_background,ydata_avg],fitted_pars=[fitted_pars_background,fitted_pars],**exp_pars)

def rabi(x, amp,period,phase,offset):
    return amp*np.cos(2*pi*x/period+phase)+offset

def ramsey(x,amp,f,phase,tau,offset):
    return amp*np.cos(2*pi*f*x+phase)*np.exp(-x/tau)+offset

def beats(x,amp,f1,f2,phase1,phase2,tau,offset):
    return amp*np.cos(pi*(f1+f2)*x+phase1)*np.cos(pi*(f2-f1)*x+phase2)*np.exp(-x/tau)+offset

def decay(x,amp,tau,offset):
    return amp*np.exp(-x/tau)+offset

def extract_freq(t_vector,y_vector,dt,plot=0):
    N = len(t_vector)
    dt = dt*1e6
    yf = scy.fft.fft(y_vector-np.mean(y_vector))
    xf = scy.fft.fftfreq(N,dt)[:round(N/2)]
    # print(len(xf))
    psd = 2.0/N * np.abs(yf[:round(N/2)])
    # print(len(psd))
    # print(psd)
    index_max = np.argmax(psd)
    if plot == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xf,psd)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power')
    # print(index_max)

    return xf[index_max]

def convert_V_to_dBm(data):

    return 10*np.log10(1e3*data**2/50)

def plot_single_shot(data_OFF,data_pi):
    states = []
    [states.append('|g>') for i in range(len(data_OFF.real))]
    [states.append('|e>') for i in range(len(data_pi.real))]
    data = {
            'I (mV)':   np.hstack((data_OFF.real*1e3,data_pi.real*1e3)),
            'Q (mV)':   np.hstack((data_OFF.imag*1e3,data_pi.imag*1e3)),
            'states':   states
                }
    dataF = pd.DataFrame(data=data,dtype=float)
    sns.jointplot(data=dataF, x='I (mV)',y='Q (mV)',hue='states')
