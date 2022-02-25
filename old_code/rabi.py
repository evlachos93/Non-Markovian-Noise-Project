# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 12:29:02 2021

@author: Evangelos

DESCRIPTION: Script for Rabi measurements (single Rabi and sweeps)
"""


import time
import importlib.util
import sys, os
sys.path.append('C:/Users/LFL/lflPython/VISAdrivers')
import smf100a as smf
import LO845m as LO
import numpy as np
import UHFQA as qa
import HDAWG as hd
import experiment_funcs as expf
import matplotlib.pyplot as plt
import csv

import scipy as scy
import plot_functions as pf 
import h5py


'''Instruments and connection'''
qa_id = 'dev2528'
awg_id = 'dev8233'
qubitLO_IP = "TCPIP::192.168.1.122::INSTR"
readoutLO_IP = "TCPIP::192.168.1.121::INSTR"
acStarkLO_IP = "USB0::0x03EB::0xAFFF::621-03A100000-0520::0::INSTR"
qubitLO = smf.SMF(qubitLO_IP,True)
readoutLO = smf.SMF(readoutLO_IP,True)
acStarkLO = LO.LO(acStarkLO_IP,True)
qubitLO.RF_ON()
readoutLO.RF_ON()
acStarkLO.RF_ON()
qubitLO.set_level(21)
readoutLO.set_freq(5.7991)
readoutLO.set_level(25)
acStarkLO.set_freq(5.7)

'''Initialize connection with Zurich Instruments'''
daq, device_qa = qa.create_api_sessions_uhf('dev2528', use_discovery= 1, ip='127.0.0.1')
awg, device_awg = hd.create_api_sessions_hd('dev8233', use_discovery= 1, ip = '127.0.0.1')

'''Channel offsets'''
awg.setDouble('/dev8233/sigouts/0/offset',0.001252)
awg.setDouble('/dev8233/sigouts/1/offset',0.008671)
awg.setDouble('/dev8233/sigouts/2/offset',0.01569)
daq.setDouble('/dev2528/sigouts/0/offset',-0.0724639893)