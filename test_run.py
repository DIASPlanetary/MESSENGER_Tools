#!/usr/bin/env python3

#File to test code in terminal to get B field time series and trajectories
#Print Mission time and add check start and end time is within mission time. 

import MESSENGER_Boundary_ID
import time

def start_prompt():
    start = input("Enter start time (YYYY-MM-DD-HH-mm-SS): ")
    try:
        time.strptime(start, "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        print("Incorrect date format. Please format as YYYY-mm-dd-HH-MM-SS")
        start=start_prompt()
    return start

def end_prompt():
    end = input("Enter end time (YYYY-mm-dd-HH-MM-SS): ")
    try:
        time.strptime(end, "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        print("Incorrect date format. Please format as YYYY-mm-dd-HH-MM-SS")
        end=end_prompt()
    return end

if __name__ == "__main__":
    print("Welcome to MUPT terminal!")
    print('Please select dates between 2011-03-23 & 2015-08-30')
    start = start_prompt()
    end = end_prompt()
    crossings = input("Which crossings do you want to plot? ([None]/philpott/sun/both):")

    if crossings == 'philpott':
        MESSENGER_Boundary_ID.mag_time_series(start,end,philpott=True)
    elif crossings == 'sun':
        MESSENGER_Boundary_ID.mag_time_series(start,end,sun=True)
    elif crossings  == 'both':
        MESSENGER_Boundary_ID.mag_time_series(start,end,sun=True,philpott=True)
    else:
        MESSENGER_Boundary_ID.mag_time_series(start,end)