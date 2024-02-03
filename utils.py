"""
Utility functions for Non-Markovian Noise Project.
"""

# Import necessary libraries
from glob import glob
import os
import csv

# Define utility functions
def get_latest_file(meas_device):
    """
    Get the latest file in the directory for the specified measurement device.
    """
    try:
        directory = 'E:\\generalized-markovian-noise\\%s\\sweep_data\\spectroscopy\\' %(meas_device)
        latest_file = max(glob.glob(os.path.join(directory, '*')), key=os.path.getmtime)
        iteration_spec = int(latest_file[-3:].lstrip('0')) + 1
    except:
        iteration_spec = 1
        
    return iteration_spec

def save_data(data, meas_device, options, experiment, iteration):
    """
    Save the data to a file.
    """

    with open("E:\\generalized-markovian-noise\\%s\\sweep_data\\%s\\%s_data_%03d.csv"%(meas_device,experiment,experiment,iteration),"w",newline="") as datafile:
        writer = csv.writer(datafile)
        writer.writerow(options.keys())
        writer.writerow(options.values())
        for i in range(len(data)):
            writer.writerow(data[i])

# Main function (if applicable)
if __name__ == "__main__":
    pass