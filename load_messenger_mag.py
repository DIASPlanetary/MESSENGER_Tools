#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:41:24 2024

@author: bowersch
"""

import numpy as np
import pandas as pd

import datetime as datetime


def get_mercury_distance_to_sun(date):
    # create a PyEphem observer for the Sun

    import ephem

    j = ephem.Mercury()

    j.compute(date, epoch='1970')

    distance_au = j.sun_distance

    return distance_au


def get_aberration_angle(date):

    import numpy as np

    r = get_mercury_distance_to_sun(date)*1.496E11

    a = 57909050*1000.

    M = 1.9891E30

    G = 6.67430E-11

    v = np.sqrt(G*M*(2./r-1./a))

    alpha = np.arctan(v/400000)

    return alpha


def get_day_of_year(date_string):
    import datetime
    date_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d-%H-%M-%S')
    return date_obj.timetuple().tm_yday


def load_MESSENGER_into_tplot(date_string, res="01", full=False, FIPS=False):
    # res can be 01,05,10,60
    doy = get_day_of_year(date_string)
    month = date_string[5:7]
    year = date_string[2:4]

    year_full = date_string[0:4]
    if doy < 10:
        doy_s = '00'+str(doy)

    elif (doy < 100) & (doy >= 10):
        doy_s = '0'+str(doy)

    else:
        doy_s = str(doy)

    # file='/Users/bowersch/Desktop/MESSENGER Data/mess-mag-calibrated avg/MAGMSOSCIAVG'+year+str(doy)+'_'+res+'_V08.TAB'

    # Data saved on the local machine, change this to ones own machine
    file = '/home/adam/Desktop/DIAS/MESSENGER/MESSENGER_Boundary_Testing/mess-mag-calibrated/' + \
        month+'/'+'MAGMSOSCIAVG'+year+doy_s+'_'+res+'_V08.TAB'
    if full == True:
        file = '/home/adam/Desktop/DIAS/MESSENGER/MESSENGER_Boundary_Testing/mess-mag-calibrated/MAGMSOSCI' + \
            year+str(doy)+'_V08.TAB'
    df = np.genfromtxt(file)

    hour = df[:, 2]

    # print(hour[0])

    minute = df[:, 3]

    second = df[:, 4]

    year = date_string[0:4]

    doy = int(doy_s)-1

    date = datetime.datetime(year=int(year), month=1,
                             day=1)+datetime.timedelta(doy)

    # print(date)

    date2 = []

    for i in range(np.size(hour)):
        if int(hour[i])-int(hour[i-1]) < 0:

            doy = doy+1

            date = datetime.datetime(
                year=int(year), month=1, day=1)+datetime.timedelta(doy)

        date2.append(
            date+datetime.timedelta(hours=hour[i], minutes=minute[i], seconds=second[i]))

    # print(date2[0])

    # time=[d.strftime("%Y-%m-%d %H:%M:%S") for d in date2]

    # print(time[0])

    time = date2

    # time=[d.timestamp for d in date2]

    # return time

    # Get B
    mag1 = df[:, 10:13]

    # Get ephemeris data
    eph = df[:, 7:10]

    ephx = df[:, 7]
    ephy = df[:, 8]
    ephz = df[:, 9]

    # Offset due to dipole field!

    ephz = ephz-479

    # Aberration:

    phi = get_aberration_angle(date_string)

    new_magx = mag1[:, 0]*np.cos(phi)-mag1[:, 1]*np.sin(phi)

    new_magy = mag1[:, 0]*np.sin(phi)+mag1[:, 1]*np.cos(phi)

    mag1[:, 0] = new_magx
    mag1[:, 1] = new_magy

    new_ephx = ephx*np.cos(phi)-ephy*np.sin(phi)

    new_ephy = ephx*np.sin(phi)+ephy*np.cos(phi)

    ephx = new_ephx
    ephy = new_ephy

    eph = np.transpose(np.vstack((ephx, ephy, ephz)))

    if full == True:
        mag1 = df[:, 9:]
        eph = df[:, 5:8]
        ephx = df[:, 5]
        ephy = df[:, 6]
        ephz = df[:, 7]

    # Define magnetic field amplitude

    magamp = np.sqrt(mag1[:, 0]**2+mag1[:, 1]**2+mag1[:, 2]**2)

    dataframe = pd.DataFrame({"Time": time})
    dataframe["mag_x"]=mag1[:,0].tolist()
    dataframe["mag_y"]=mag1[:,1].tolist()
    dataframe["mag_z"]=mag1[:,2].tolist()
    dataframe["magamp"]=magamp.tolist()
    dataframe["eph_x"]=ephx.tolist()
    dataframe["eph_y"]=ephy.tolist()
    dataframe["eph_z"]=ephz.tolist()

    return dataframe
    #return time, mag1, magamp, eph

    