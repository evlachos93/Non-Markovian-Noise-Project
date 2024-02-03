#!/usr/bin/env python
# coding: utf-8

# # Import

# In[58]:


import textwrap
import json
import urllib
import jsonschema
import zhinst.ziPython as zp
import zhinst.utils as zu
import zhinst.toolkit as zt
import time
import numpy as np
import UHFQA as qa


# # Predefined functions

# In[2]:

# Please refer to https://docs.zhinst.com/hdawg/commandtable/v2/schema for other settings


def ct_pulse_length(n_wave, pulse_length_start = 32, pulse_length_increment = 16, sequence='rabi'):
    """
    Creates a command table by sweeping the duration of the pulse

    Args:
        n_wave (int): Number of pulses
        pulse_length_start (int): Minimum pulse duration in awg units. Must be larger than 32 and multiple of 16
        pulse_length_increment (int): Pulse length increment in awg units. Must be multiple of 16, otherwise it is padded with zeros at the end, leading to timing issues
        
    Returns:
        ct (dict): Dictionary containing command table instructions
    """
    ct = {'header':{'version':'0.2'}, 'table':[]}

    for i in range(n_wave):
        sample_length = pulse_length_start + i * pulse_length_increment
        entry = {'index': i,
                  'waveform':{
                      'index': 0,
                      'length':  sample_length
                  },
                  }

        ct['table'].append(entry)

    return ct



# Please refer to https://docs.zhinst.com/hdawg/commandtable/v2/schema for other settings
def ct_amplitude_increment(amplitude_start=[0.5,0.5],amplitude_increment = 0.001*np.ones(2)):
    """
    Creates a command table where the amplitude of a pulse is swept

    Args:
        amplitude_start (list, optional): List containing initial amplitudes for the two waveforms. Defaults to [0.5,0.5].
        amplitude_increment (array, optional): Pulse amplitude increment. Defaults to 0.001*np.ones(2).

    Returns:
        ct (dict): Dictionary containing command table instructions
    """

    ct = {'header':{'version':'0.2'}, 'table':[]}

    entry = {'index': 0,
               # 'waveform':{
               #     'index': 0
               # },
             'amplitude0':{
                 'value':amplitude_start[0],
                 'increment': False
             },
             'amplitude1':{
                 'value':amplitude_start[1],
                 'increment': False
             }
            }
    ct['table'].append(entry)

    # second entry defines increment and waveform
    entry = {'index': 1,
              'waveform':{
                       'index': 0,
                       'length': 1024
              },
             'amplitude0':{
                 'value':amplitude_increment[0],
                 'increment': True
             },
             'amplitude1':{
                 'value':amplitude_increment[1],
                 'increment': True
             }
            }

    ct['table'].append(entry)
    return ct
