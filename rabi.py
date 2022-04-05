# -*- coding: utf-8 -*-

'''----------------------------------------------------------Rabi---------------------------------------------------------'''
from configuration import *

A_d = 0.255
try:
    directory = 'E:\\generalized-markovian-noise\\%s\\sweep_data\\rabi\\' %(meas_device)
    latest_file = max(glob.glob(os.path.join(directory, '*')), key=os.path.getmtime)
    iteration_rabi = int(latest_file[-3:].lstrip('0')) + 1
except:
    iteration_rabi = 1

options_rabi = {
    'sampling_rate':    2.4e9,
    'qubitDriveFreq':   3.209e9,
    'integration_length':   2.3e-6,
    'prePulseLength':   850e-9,
    'postPulseLength':  200e-9,
    'cav_resp_time':    0.25e-6,
    'nAverages':        256,
    'stepSize':         6e-9,
    'Tmax':             0.6e-6,
    'amplitude_hd':     A_d,
    'sequence':         'rabi',
    'measPeriod':       600e-6,
    'AC_pars':          [0,0],
    'AC_freq':          7.3586e9,
    }

if options_rabi['AC_pars'] == 0:
    acStarkLO.setRFOn(bRFOn=False)
else:
    acStarkLO.setRFOn(bRFOn=True)
    acStarkLO.setFrequency(options_rabi['AC_freq'])

qubitLO.set_freq(options_rabi['qubitDriveFreq']/1e9)

t,I,Q,nPoints = expf.pulse(daq,awg,**options_rabi)

# plot data
# options_rabi['AC_pars'][0] = options_rabi['AC_pars'][0]*6/20
data = Q
fitted_pars,error = pf.fit_data(x_vector=t,y_vector=data,dt=t[-1]/nPoints,**options_rabi)
pf.plot_data(awg,x_vector=t,y_vector=data,fitted_pars=fitted_pars,**options_rabi,iteration=iteration_rabi)

pi_pulse = np.round(1/2*fitted_pars[1])
threshold = round(np.mean(data)*2**12)

# save data
with open("E:\\generalized-markovian-noise\\%s\\rabi\\%s_data_%03d.csv"%(meas_device,'rabi',iteration_rabi),"w",newline="") as datafile:
    writer = csv.writer(datafile)
    writer.writerow(options_rabi.keys())
    writer.writerow(options_rabi.values())
    writer.writerow(t)
    writer.writerow(I)

iteration_rabi += 1
