#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:33:16 2024

@author: bowersch
"""

from scipy.signal import find_peaks
import tqdm
import MESSENGER_Boundary_ID as mag
import load_messenger_mag as load_messenger_mag
import pickle
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ephem
import datetime

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                                                    '#f781bf', '#a65628', '#984ea3',
                                                    '#999999', '#e41a1c', '#dede00'])

'''
Load list if already saved in pickle
'''
def start():
    ''' Loads in crossing lists and returns data frames:
    df_all -- 10 minute resolution ephemeris data for entire mission,
    df_p -- Philpott crossings list with orbit number
    df_sun -- Sun crossings list with orbit number
    '''
    with open('df_p.pickle', 'rb') as f:
        df_p = pickle.load(f)
    with open('df_s.p', 'rb') as f:
        df_sun = pickle.load(f)
    # df_all = read_in_all()
    df_all = pd.read_pickle("df_all.pkl")
    df_p = orbit(df_all, df_p)
    df_sun = orbit(df_all, df_sun)
    return df_all, df_p, df_sun


def info():
    ''' Load in data frame with ephemeris 
    and mag data for the entire mission at 1 second resolution.
    Need to create pickle first, see line 2049
    '''
    df_info = pd.read_pickle("df_info_all.pkl")
    return df_info

''' Code to load in MESSENGER boundaries identified by Philpott and Sun.

 Link to download Philpott boundary list:  
    
     https://doi.org/10.1029/2019JA027544
    
   Want the jgra55678-sup-0002-Table_SI-S01.xlsx file in the 
   Supplementary Material section. 

   Then, save this file as a .csv file for more easy use in python


 Specify location on you machine of this file here:
    '''


philpott_file = '/home/adam/Desktop/DIAS/MESSENGER/MESSENGER_Boundary_Testing/jgra55678-sup-0002-table_si-s01.csv'


''' Link to download Sun boundary list: 
    
     https://zenodo.org/records/8298647
    
    Need to download separate .txt files for each boundary crossing type.
    
    Save these into a folder on your machine.

    Specify location of folder: '''

Sun_crossings_folder = '/home/adam/Desktop/DIAS/MESSENGER/MESSENGER_Boundary_Testing/Weijie Crossings'


'''

The Sun crossing list only has the times of the boundaries, not their location.

I have included a .csv file in the github that has both the boundary times and locations 
of the Sun list (Sun_Boundaries_with_Eph.csv).

Specify the location of this .csv file here:

'''

Sun_file = '/home/adam/Desktop/DIAS/MESSENGER/MESSENGER_Boundary_Testing/Sun_Boundaries_with_Eph.csv'


'''
 Will need to install the ephem package in python to calculate Mercury-Sun distance,
 which is required to rotate into the aberrated Mercury Solar Magnetospheric (MSM')
 coordinate frame
 
 pip install ephem
 
 
 Examples: 
     
     To plot locations of all boundaries identified by the Philpott list:
     
         philpott_file = 'PHILPOTT FILE LOCATION'
     
        df_p = read_in_Philpott_list(philpott_file)
     
        plot_boundary_locations(df_p)
        
    To plot locations of all boundaries identified by the Sun list:
        
        Sun_file = 'SUN CSV FILE LOCATION'
        
        df_Sun = read_in_Sun_csv(Sun_file)
        
        plot_boundary_locations(df_Sun)     
       
'''



def convert_to_datetime(date_string):
    ''' converts date_string to datetime object'''
    date_obj = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

    return date_obj


def get_mercury_distance_to_sun(date):
    # create a PyEphem observer for the Sun

    j = ephem.Mercury()
    j.compute(date, epoch='1970')
    distance_au = j.sun_distance

    return distance_au


def read_in_Philpott_list(pf):
    ''' Create a dataframe of boundary crossings as identified by the Philpott List

    This list also provided ephemeris coordinates for the boundaries, so we also include
    these parameters in the dataframe, but rotate them to be in the aberrated MSM' coordinate 
    system.

    Input:

        String of the location for the philpott .csv file on your machine

    Output:

        Dataframe of boundary crossings with a start and end time, start and end 
        location in the MSM' coordinate system, and the type of crossing.

        mp_in = inbound magnetopause

        bs_in = inbound bow shock

        bs_out = outbound bow shock

        mp_out = outbound magnetopause

        gap = data gap locations


    Example: df_philpott = read_in_Philpott_list(philpott_file)

    '''

    filename = pf

    df_boundaries = pd.read_csv(filename)

    def create_datestring(year, day_of_year, hour, minute, second):
        # Create a datetime object for January 1st of the given year
        date = datetime.datetime(int(year), 1, 1)

        # Add the number of days (day_of_year - 1) to get to the desired date
        date += datetime.timedelta(days=float(day_of_year) - 1, hours=float(
            hour), minutes=float(minute), seconds=float(second))

        return date

    dt = np.array([create_datestring(df_boundaries.Yr_pass.iloc[p],
                                     df_boundaries.Day_orbit.iloc[p],
                                     df_boundaries.Hour.iloc[p],
                                     df_boundaries.Minute.iloc[p],
                                     round(df_boundaries.Second.iloc[p]))
                   for p in range(len(df_boundaries))])

    df_boundaries['time'] = dt

    Z_MSM = df_boundaries['Z_MSO (km)']/2440-.19

    X_MSM = np.array([])

    Y_MSM = np.array([])

    cross_string = np.array([])

    cross_strings = np.array(['err', 'bs_in_1', 'bs_in_2', 'mp_in_1', 'mp_in_2',
                              'mp_out_1', 'mp_out_2', 'bs_out_1', 'bs_out_2', 'gap_1', 'gap_2'])

    def rotate_into_msm(x, y, z, time):

        # Aberration:

        def get_aberration_angle(date):

            # Estimate instantaneous orbital velocity of Mercury:

            r = get_mercury_distance_to_sun(date)*1.496E11

            a = 57909050*1000.

            M = 1.9891E30

            G = 6.67430E-11

            v = np.sqrt(G*M*(2./r-1./a))

            # Calculate aberration angle assuming 400 km/s sw speed

            alpha = np.arctan(v/400000)

            return alpha

        phi = get_aberration_angle(time)

        x_msm = x*np.cos(phi)-y*np.sin(phi)

        y_msm = y*np.sin(phi)+y*np.cos(phi)

        return x_msm, y_msm, z

    for i in range(len(df_boundaries)):

        X_MSM_1, Y_MSM_1, Z_MSM_2 = rotate_into_msm(df_boundaries['X_MSO (km)'].iloc[i]/2440,
                                                    df_boundaries['Y_MSO (km)'].iloc[i]/2440,
                                                    Z_MSM[i],
                                                    df_boundaries.time.iloc[i])

        X_MSM = np.append(X_MSM, X_MSM_1)

        Y_MSM = np.append(Y_MSM, Y_MSM_1)

        cross_string = np.append(
            cross_string, cross_strings[df_boundaries['Boundary number'].iloc[i]])

    df_boundaries[['X_MSM', 'Y_MSM', 'Z_MSM']] = np.stack(
        (X_MSM, Y_MSM, Z_MSM), axis=1)

    df_boundaries['Cross_Type'] = cross_string

    stacked_df_data = pd.DataFrame({'start': [np.nan],
                                    'end': [np.nan],
                                    'start_x_msm': [np.nan],
                                    'end_x_msm': [np.nan],
                                    'start_y_msm': [np.nan],
                                    'end_y_msm': [np.nan],
                                    'start_z_msm': [np.nan],
                                    'end_z_msm': [np.nan],
                                    'Type': [np.nan]})

    cross_strings = ['bs_in', 'bs_out', 'mp_in', 'mp_out', 'gap']

    for i in cross_strings:

        s = df_boundaries[(df_boundaries.Cross_Type == i+'_1')]
        e = df_boundaries[(df_boundaries.Cross_Type == i+'_2')]

        data = {'start': s.time.to_numpy(),
                'end': e.time.to_numpy(),
                'start_x_msm': s.X_MSM.to_numpy(),
                'end_x_msm': e.X_MSM.to_numpy(),
                'start_y_msm': s.Y_MSM.to_numpy(),
                'end_y_msm': e.Y_MSM.to_numpy(),
                'start_z_msm': s.Z_MSM.to_numpy(),
                'end_z_msm': e.Z_MSM.to_numpy(),
                'Type': i}

        stacked_df_data = pd.concat([stacked_df_data, pd.DataFrame(data)])

    # stacked_df_data = stacked_df_data.drop(0).reset_index(drop=True)

    stacked_df_data = stacked_df_data.sort_values('start', ignore_index=True)

    stacked_df_data = stacked_df_data.dropna()

    return stacked_df_data


def read_in_Sun_files(scf):
    ''' Input the path for the Sun_crossings_folder

    Outputs a dataframe of all the crossings, with a row for start of 
    crossing interval, the end of crossing interval, and the type of crossing:

        mp_in = inbound magnetopause

        bs_in = inbound bow shock

        bs_out = outbound bow shock

        mp_out = outbound magnetopause

    Example: Sun_crossings = read_in_Sun_files(Sun_crossings_folder)

    '''
    def convert_Sun_txt_to_date(file):
        ''' Convert the Sun files to a date_string YYYY-MM-DD HH:MM:SS '''

        x_in = np.loadtxt(file, usecols=(0, 1, 2, 3, 4, 5))
        x_out = np.loadtxt(file, usecols=(6, 7, 8, 9, 10, 11))
        date_in = np.array([])
        date_out = np.array([])

        # Correct for annoying run overs (minutes = 60, etc.)
        for i in range(np.size(x_in[:, 0])):

            if int(np.floor(x_in[i, 5])) >= 60:
                x_in[i, 5] = 0.0
                x_in[i, 4] = x_in[i, 4]+1

            if int(np.floor(x_out[i, 5])) >= 60:
                x_out[i, 5] = 0.0
                x_out[i, 4] = x_out[i, 4]+1

            if int(np.floor(x_out[i, 5])) < 0:
                x_out[i, 5] = 59
                x_out[i, 4] = x_out[i, 4]-1

                if x_out[i, 4] < 0:
                    x_out[i, 3] = x_out[i, 3]-1
                    x_out[i, 4] = 59

            if int(np.floor(x_in[i, 5])) < 0:
                x_in[i, 5] = 59
                x_in[i, 4] = x_in[i, 4]-1
                if x_in[i, 4] < 0:
                    x_in[i, 3] = x_in[i, 3]-1
                    x_in[i, 4] = 59

            def convert_to_datetime(date_string):
                date_obj = datetime.datetime.strptime(
                    date_string, "%Y-%m-%d %H:%M:%S")

                return date_obj

            date_string_in = str(int(np.floor(x_in[i, 0])))+'-'+str(int(np.floor(x_in[i, 1]))) +\
                '-'+str(int(np.floor(x_in[i, 2])))+' '+str(int(np.floor(x_in[i, 3]))) +\
                ':'+str(int(np.floor(x_in[i, 4]))) + \
                ':'+str(int(np.floor(x_in[i, 5])))

            date_datetime_in = convert_to_datetime(date_string_in)

            date_in = np.append(date_in, date_datetime_in)

            date_string_out = str(int(np.floor(x_out[i, 0])))+'-'+str(int(np.floor(x_out[i, 1]))) +\
                '-'+str(int(np.floor(x_out[i, 2])))+' '+str(int(np.floor(x_out[i, 3]))) +\
                ':'+str(int(np.floor(x_out[i, 4]))) + \
                ':'+str(int(np.floor(x_out[i, 5])))

            date_datetime_out = convert_to_datetime(date_string_out)

            date_out = np.append(date_out, date_datetime_out)

        date = np.array([date_in, date_out])

        return date

    file_mp_in = scf+'MagPause_In_Time_Duration__public_version_WeijieSun_20230829.txt'
    file_mp_out = scf+'MagPause_Out_Time_Duration_public_version_WeijieSun_20230829.txt'
    file_bs_in = scf+'Bow_Shock_In_Time_Duration__public_version_WeijieSun_20230829.txt'
    file_bs_out = scf+'Bow_Shock_Out_Time_Duration_public_version_WeijieSun_20230829.txt'

    mp_in = convert_Sun_txt_to_date(file_mp_in)
    mp_out = convert_Sun_txt_to_date(file_mp_out)
    bs_in = convert_Sun_txt_to_date(file_bs_in)
    bs_out = convert_Sun_txt_to_date(file_bs_out)

    def generate_crossing_dataframe(cross, typ, eph=False):

        cross_start = cross[0, :]

        cross_end = cross[1, :]

        cross_df = pd.DataFrame(data={'start': cross_start, 'end': cross_end})

        cross_df['Type'] = typ

        return cross_df

    mi = generate_crossing_dataframe(mp_in, 'mp_in')

    mo = generate_crossing_dataframe(mp_out, 'mp_out')

    bi = generate_crossing_dataframe(bs_in, 'bs_in')

    bo = generate_crossing_dataframe(bs_out, 'bs_out')

    crossings = [mi, mo, bi, bo]

    cc = pd.concat(crossings)

    c = cc.sort_values('start')

    return c


def read_in_Sun_csv(Sun_csv):

    df_Sun = pd.read_csv(Sun_csv)

    start = np.array([convert_to_datetime(d) for d in df_Sun.start])
    end = np.array([convert_to_datetime(d) for d in df_Sun.end])

    df_Sun['start'] = start

    df_Sun['end'] = end

    return df_Sun


def plot_boundary_locations(df):
    '''Create a plot of Mercury and the location of the boundaries in cylindrical coordinates

    Input a dataframe loaded by read_in_Philpott_list or read_in_Sun_list

    Outputs a plot of the magnetopause and bow shock boundaries onto a cylindrical map of
    Mercury with dashed lines for the nominal magnetopause and bow shock shapes determined
    by Winslow et al., 2013

    Colours of magnitopause based on Mercury's distance to the sun
    '''

    # Plot Mercury

    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)-0.2

    fig, ax1 = plt.subplots(1)

    # Plot the circle in all 3 plots
    ax1.plot(x, y, color='gray')

    ax1.set_xlabel("$X_{MSM\'}$ ($R_M$)", fontsize=20)

    ax1.set_ylabel("\u03C1$_{MSM\'}$ ($R_M$)", fontsize=20)

    ax1.tick_params(labelsize=20)

    def plot_mp_and_bs(ax1):
        ''' Plot Nominal Magnetopause and Bow Shock Location from Winslow 2013'''

        y_mp = np.linspace(-100, 100, 100)
        z_mp = np.linspace(-100, 100, 100)
        x_mp = np.linspace(-10, 10, 100)

        rho = np.sqrt(y_mp**2+(z_mp)**2)

        phi = np.arctan2(rho, x_mp)

        Rss = 1.45

        alpha = 0.5

        phi2 = (np.linspace(0, 2*np.pi, 100))

        rho = Rss*(2/(1+np.cos(phi2)))**(alpha)

        xmp = rho*np.cos(phi2)

        ymp = rho*np.sin(phi2)

        ax1.plot(xmp, ymp, color='black', linestyle='--', linewidth=3)

        psi = 1.04

        p = 2.75

        L = psi*p

        x0 = .5

        phi = (np.linspace(0, 2*np.pi, 100))
        rho = L/(1. + psi*np.cos(phi))

        xshock = x0 + rho*np.cos(phi)
        yshock = rho*np.sin(phi)

        ax1.plot(xshock, yshock, color='black', linestyle='--', linewidth=3)

    plot_mp_and_bs(ax1)

    # Color the left hemisphere red and the right hemisphere gray
    ax1.fill_between(x, y, where=x < 0, color='black', interpolate=True)
    # Set equal aspect so Mercury is circular
    ax1.set_aspect('equal', adjustable='box')

    # Set the limits of the plot in all 3 plots
    ax1.set_xlim([-5, 3])
    ax1.set_ylim([0, 4])

    df_mp = df[((df.Type == 'mp_in') | (df.Type == 'mp_out'))]

    df_bs = df[((df.Type == 'bs_in') | (df.Type == 'bs_out'))]

    def plot_mean_locations(df, cr, lb):

        mean_x = np.mean(df[['start_x_msm', 'end_x_msm']], axis=1)

        mean_y = np.mean(df[['start_y_msm', 'end_y_msm']], axis=1)

        mean_z = np.mean(df[['start_z_msm', 'end_z_msm']], axis=1)

        r_msm = np.sqrt(mean_y**2+mean_z**2)

        ax1.scatter(mean_x, r_msm, s=.1, color=cr, label=lb)

    plot_mean_locations(df_mp, 'indianred', 'MP')
    plot_mean_locations(df_bs, 'mediumturquoise', 'BS')

    ax1.legend()


def plot_boundary_locations_solar_distance(df):
    '''Plotting the MagnetoPause colour coded based on
        distance mercury is from sun during measurement.
        df -- crossing list data frame
    '''

    df_mp = df[((df.Type == 'mp_in') | (df.Type == 'mp_out'))]

    df_bs = df[((df.Type == 'bs_in') | (df.Type == 'bs_out'))]

    # Adding Average Date and Mercury distance to sun to dataframe
    def AvgDate_Distance(df):
        avg_date = (df[['start', 'end']].mean(axis=1))
        df = df.assign(AvgDate=avg_date)
        distance = df['AvgDate'].apply(
            lambda date: get_mercury_distance_to_sun(date))
        df = df.assign(Distance=distance)
        return df

    df_mp = AvgDate_Distance(df_mp)

    df_bs = AvgDate_Distance(df_bs)

    # Dividing MP into 4 distance regions
    numBin = 4
    MPmin = df_mp["Distance"].min()
    MPmax = df_mp["Distance"].max()
    MPBins = (MPmax-MPmin)/numBin

    # Create new dataframes based on distances
    min = MPmin
    Bins = MPBins

    df_mp1 = df_mp[df_mp["Distance"] < min+Bins]
    df_mp2 = df_mp[(df_mp["Distance"] > min+Bins) &
                   (df_mp["Distance"] < min+2*Bins)]
    df_mp3 = df_mp[(df_mp["Distance"] > min+2*Bins) &
                   (df_mp["Distance"] < min+3*Bins)]
    df_mp4 = df_mp[(df_mp["Distance"] > min+3*Bins) &
                   (df_mp["Distance"] < min+4*Bins)]

    # Dividing BS into 4 distance regions
    numBin = 4
    BSmin = df_bs["Distance"].min()
    BSmax = df_bs["Distance"].max()
    BSBins = (BSmax-BSmin)/numBin

    # Create new dataframes based on distances
    min = BSmin
    Bins = BSBins
    df_bs1 = df_bs[df_bs["Distance"] < min+Bins]
    df_bs2 = df_bs[(df_bs["Distance"] > min+Bins) &
                   (df_bs["Distance"] < min+2*Bins)]
    df_bs3 = df_bs[(df_bs["Distance"] > min+2*Bins) &
                   (df_bs["Distance"] < min+3*Bins)]
    df_bs4 = df_bs[(df_bs["Distance"] > min+3*Bins) &
                   (df_bs["Distance"] < min+4*Bins)]

    # Plot Mercury

    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)-0.2

    fig, ax1 = plt.subplots(1)

    # Plot the circle in all 3 plots
    ax1.plot(x, y, color='gray')

    ax1.set_xlabel("$X_{MSM\'}$ ($R_M$)", fontsize=20)

    ax1.set_ylabel("\u03C1$_{MSM\'}$ ($R_M$)", fontsize=20)

    ax1.tick_params(labelsize=20)

    def plot_mp_and_bs(ax1):
        ''' Plot Nominal Magnetopause and Bow Shock Location from Winslow 2013'''

        y_mp = np.linspace(-100, 100, 100)
        z_mp = np.linspace(-100, 100, 100)
        x_mp = np.linspace(-10, 10, 100)

        rho = np.sqrt(y_mp**2+(z_mp)**2)

        phi = np.arctan2(rho, x_mp)

        Rss = 1.45

        alpha = 0.5

        phi2 = (np.linspace(0, 2*np.pi, 100))

        rho = Rss*(2/(1+np.cos(phi2)))**(alpha)

        xmp = rho*np.cos(phi2)

        ymp = rho*np.sin(phi2)

        ax1.plot(xmp, ymp, color='black', linestyle='--', linewidth=3)

        psi = 1.04

        p = 2.75

        L = psi*p

        x0 = .5

        phi = (np.linspace(0, 2*np.pi, 100))
        rho = L/(1. + psi*np.cos(phi))

        xshock = x0 + rho*np.cos(phi)
        yshock = rho*np.sin(phi)

        ax1.plot(xshock, yshock, color='black', linestyle='--', linewidth=3)

    plot_mp_and_bs(ax1)

    # Color the left hemisphere red and the right hemisphere gray
    ax1.fill_between(x, y, where=x < 0, color='black', interpolate=True)
    # Set equal aspect so Mercury is circular
    ax1.set_aspect('equal', adjustable='box')

    # Set the limits of the plot in all 3 plots
    ax1.set_xlim([-5, 3])
    ax1.set_ylim([0, 4])

    def plot_mean_locations(df, cr, lb):

        mean_x = np.mean(df[['start_x_msm', 'end_x_msm']], axis=1)

        mean_y = np.mean(df[['start_y_msm', 'end_y_msm']], axis=1)

        mean_z = np.mean(df[['start_z_msm', 'end_z_msm']], axis=1)

        r_msm = np.sqrt(mean_y**2+mean_z**2)

        if lb == 'MP':
            more = df["Distance"].min()
            less = df["Distance"].max()

            ax1.scatter(mean_x, r_msm, s=.1, color=cr,
                        label=f'MP ({more:.2f} < D < {less:.2f}) (au)')

        else:
            more = df["Distance"].min()
            less = df["Distance"].max()

            ax1.scatter(mean_x, r_msm, s=.1, color=cr,
                        label=f'BS ({more:.2f} < D < {less:.2f}) (au)')
        # else:
        #     ax1.scatter(mean_x, r_msm,s=.1,color=cr,label = lb)

    # creating a colourmap
    import matplotlib.cm as cm

    colours = cm.rainbow(np.linspace(0, 1, numBin))

    plot_mean_locations(df_mp1, 'red', 'MP')
    plot_mean_locations(df_mp2, 'green', 'MP')
    plot_mean_locations(df_mp3, 'black', 'MP')
    plot_mean_locations(df_mp4, 'blue', 'MP')
    plot_mean_locations(df_bs, 'mediumturqoise', 'BS')
    # plot_mean_locations(df_bs1, 'mediumturquoise', 'BS')
    # plot_mean_locations(df_bs2, 'mediumturquoise', 'BS')
    # plot_mean_locations(df_bs3, 'mediumturquoise', 'BS')
    # plot_mean_locations(df_bs4, 'mediumturquoise', 'BS')

    ax1.legend(fontsize="8")


def plot_boundary_over_year(df, EndYrs=18, StartYrs=0):
    '''  Plots X_MSM', rho_MSM', and distance from the 
    sun as a function of Mercury years since the start date.
    Inputs:
    df -- crossing list data frame
    EndYrs -- Mercury years from the first measurement to plot to
    StartYrs -- Mercury years from the first measurement to plot from
    '''
    array = ["mp_in", "mp_out"]
    # Make new list with only mp data
    df_mp = df.loc[df['Type'].isin(array)]

    avg_date = (df_mp[['start', 'end']].mean(axis=1))
    df_mp = df_mp.assign(AvgDate=avg_date)
    distance = df_mp['AvgDate'].apply(
        lambda date: get_mercury_distance_to_sun(date))
    df_mp = df_mp.assign(Distance=distance)

    FirstDay = df_mp["AvgDate"].min()
    df_mp["AvgDate"] = (df_mp["AvgDate"]-FirstDay)

    df_TargetYear = df_mp.loc[(df_mp["AvgDate"] < pd.Timedelta(
        days=EndYrs*88)) & (df_mp["AvgDate"] > pd.Timedelta(days=StartYrs*88))]

    fig, axs = plt.subplots(3, sharex=True)

    axs[0].set_ylabel("$X_{MSM\'}$ ($R_M$)", fontsize=10)

    axs[1].set_ylabel("\u03C1$_{MSM\'}$ ($R_M$)", fontsize=10)

    axs[2].set_ylabel("Distance from Sun (au)", fontsize=10)

    def plot_mean_locations(df, cr, lb):

        mean_x = np.mean(df[['start_x_msm', 'end_x_msm']], axis=1)

        mean_y = np.mean(df[['start_y_msm', 'end_y_msm']], axis=1)

        mean_z = np.mean(df[['start_z_msm', 'end_z_msm']], axis=1)

        r_msm = np.sqrt(mean_y**2+mean_z**2)

        # Converting time from ns to a mercury year approx 88 days
        time = df["AvgDate"].astype(int) / 7.6E15
        distance = df["Distance"].astype(float)

        axs[0].scatter(time, r_msm, s=.1, color=cr, label=lb)
        axs[1].scatter(time, mean_x, s=.1, color=cr, label=lb)
        axs[2].scatter(time, distance, s=.1, color=cr, label=lb)
        plt.xlabel("Mercury Year Since First Measurement")

    plot_mean_locations(df_TargetYear, 'red', 'MP')


def plot_distance_from_nominal(df):
    '''
    Function to plot the measured magnetopause distance from 
    the nominal magnetopause obtained by Winslow et al., 2013.
    df -- crossing list data frame 
    '''
    df_mp = df[((df.Type == 'mp_in') | (df.Type == 'mp_out'))]

    y_mp = np.linspace(-100, 100, 100)
    z_mp = np.linspace(-100, 100, 100)
    x_mp = np.linspace(-10, 10, 100)

    rho = np.sqrt(y_mp**2+(z_mp)**2)

    phi = np.arctan2(rho, x_mp)

    Rss = 1.45

    alpha = 0.5

    phi2 = (np.linspace(0, 2*np.pi, 100))

    rho = Rss*(2/(1+np.cos(phi2)))**(alpha)

    xmp = rho*np.cos(phi2)

    ymp = rho*np.sin(phi2)

    # def nominalX(phi2):
    #     return rho*np.cos(phi2)

    # def nominalY(phi2):
    #     return rho*np.sin(phi2)

    mean_x = np.mean(df_mp[['start_x_msm', 'end_x_msm']], axis=1)

    mean_y = np.mean(df_mp[['start_y_msm', 'end_y_msm']], axis=1)

    mean_z = np.mean(df_mp[['start_z_msm', 'end_z_msm']], axis=1)

    r_msm = np.sqrt(mean_y**2+mean_z**2)

    mean_x = mean_x.values
    r_msm = r_msm.values

    # Finding the distance drom the nominal
    mindist = []
    for i in range(0, len(mean_x)):
        dist = []
        for j in range(0, len(phi2)):
            xdist = (mean_x[i]-xmp[j])
            ydist = (r_msm[i]-ymp[j])
            if xdist < 0 or ydist < 0:
                dist.append(-np.sqrt(xdist**2+ydist**2))
            else:
                dist.append(np.sqrt(xdist**2+ydist**2))
        mindist.append(dist[np.argmin(np.abs(dist))])
        # print(dist[np.argmin(np.abs(dist))])
    # print(mindist)

    def AvgDate_Distance(df):
        avg_date = (df[['start', 'end']].mean(axis=1))
        df = df.assign(AvgDate=avg_date)
        distance = df['AvgDate'].apply(
            lambda date: get_mercury_distance_to_sun(date))
        df = df.assign(Distance=distance)
        return df

    df_mp = AvgDate_Distance(df_mp)

    fig, axs = plt.subplots(2, sharex=True)

    # Now plot mindist versus time
    time = df_mp["AvgDate"]
    distance = df_mp["Distance"].astype(float)

    import matplotlib.dates as mdates

    axs[0].scatter(time, mindist, s=.1)
    axs[1].scatter(time, distance, s=.1)

    # fig.legend()
    plt.xlabel("Time")
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # :%b'))
    axs[0].set_ylabel("Distance from nominal point", fontsize=7)
    axs[1].set_ylabel("Distance from Sun (au)", fontsize=10)
    # plt.xlim(15100,15300)


def mag_time_series(date_string, res="01", full=False, FIPS=False):
    '''
    Plot timeseries of B fields in 3-axis
    Inputs:
    date_string -- Date of interest 
    res -- resolution of data in seconds (options 01, 05, 10, 60)
    full -- 
    FIPS -- Not functional
    '''

    time, mag, magamp, eph = load_messenger_mag.load_MESSENGER_into_tplot(
        date_string, res, full, FIPS)

    fig, axs = plt.subplots(5, sharex=True)

    # fig.set_size_inches(10,12)

    plt.xlabel(f"Date: {date_string}")

    axs[0].set_ylabel("$B_x$ (nT)", fontsize=12)
    axs[0].plot(time, mag[:, 0], linewidth=0.8)

    axs[1].set_ylabel("$B_y$ (nT)", fontsize=12)
    axs[1].plot(time, mag[:, 1], linewidth=0.8)

    axs[2].set_ylabel("$B_z$ (nT)", fontsize=12)
    axs[2].plot(time, mag[:, 2], linewidth=0.8)

    axs[3].set_ylabel("|B| (nT)", fontsize=12)
    axs[3].plot(time, magamp, linewidth=0.8)

    axs[4].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[4].set_ylabel("(nT)", fontsize=12)
    axs[4].plot(time, mag[:, 0], linewidth=0.8, label='$B_x$')
    axs[4].plot(time, mag[:, 1], linewidth=0.8, label='$B_y$')
    axs[4].plot(time, mag[:, 2], linewidth=0.8, label='$B_z$')
    axs[4].plot(time, magamp, linewidth=0.8, label='|B|')
    axs[4].legend()


def all_mag_time_series(date_string, res="01", full=False, FIPS=False):
    '''
    Function to plot all B fields for a given time on one plot
    Inputs:
    date_string -- Date of interest 
    res -- resolution of data in seconds (options 01, 05, 10, 60)
    full -- 
    FIPS -- Not functional
    '''

    time, mag, magamp, eph = load_messenger_mag.load_MESSENGER_into_tplot(
        date_string, res, full, FIPS)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_ylabel("(nT)", fontsize=16)
    ax.plot(time, mag[:, 0], linewidth=0.8, label='$B_x$')
    ax.plot(time, mag[:, 1], linewidth=0.8, label='$B_y$')
    ax.plot(time, mag[:, 2], linewidth=0.8, label='$B_z$')
    ax.plot(time, magamp, linewidth=0.8, label='|B|')
    ax.legend()


# Breaking lists into 2 dif types, identifying difference betweem starts
# times and ploting histogram
def compare_lists(df, df_p, plot=False):
    ''' Connects every Sun crossing to the closest temporal Philpott crossing
    Order important as Sun list shorter
    Inputs:
    df -- Sun crossing data frame
    df_p -- Philpott crossing data frame
    plot -- plot histogram of difference in start times of partnered crossings (default = False)  
    
    Outputs:
    Paired bowshock and magnetopause data frames
    '''

    # df = df[df.index > 14800]
    # df_p = df_p[df_p.index < 1600]

    def split(df):
        ''' Split crossing list into magnetopause and bowshock crossings'''
        df_mp = df[(df.Type == 'mp_in') | (df.Type == 'mp_out')]
        df_bs = df[(df.Type == 'bs_in') | (df.Type == 'bs_out')]
        return df_mp, df_bs

    df_mp, df_bs = split(df)
    df_p_mp, df_p_bs = split(df_p)

    def find_partner(df, df_p):
        ''' Finds closest temporal crossing in Philpott list to given Sun crossing'''
        min_start = []
        min_end = []
        min_diff = []
        #!!!Weird behaviour here?!!!
        for i in df.start:
            df_temp = (df_p.start - i).abs()
            min_index = df_temp.idxmin()

            min_start.append(df_p.loc[min_index, 'start'])
            min_end.append(df_p.loc[min_index, 'end'])
            min_diff.append(df_temp[min_index].total_seconds())

        # print(df_temp)
        # Find this min index and append horizontally to sun df
        df["startP"] = min_start
        df['endP'] = min_end
        df['startDif'] = min_diff

        df['Flag'] = 0

        # Set 'Flag' to 1 where there is no overlap between the crossings
        x = df.index[~(((df['start'] >= df['startP'])
                        & (df['start'] <= df['endP'])) | ((df['end'] >= df['startP'])
                                                          & (df['end'] <= df['endP'])) | ((df['startP'] >= df['start'])
                                                                                          & (df['startP'] <= df['end'])) | ((df['endP'] >= df['start'])
                                                                                                                            & (df['endP'] <= df['end'])))]

        df.loc[df.index.isin(x), 'Flag'] = 1

        # Set 'Flag' to 2 where 'startDif' is greater than or equal to 3600 seconds
        df.loc[df['startDif'] >= 3600, 'Flag'] = 2
        extreme_count = df.loc[df.Flag == 2, 'Flag'].count()
        print("EXTREME:", extreme_count)

        # Set 'Flag' to 3 when orbit disagrees by just a few minutes (<10 minutes)
        df.loc[(df['startDif'] >= 600) & (df['Flag'] == 1), 'Flag'] = 3

        return df

    df_part_mp = find_partner(df_mp, df_p_mp)
    df_part_bs = find_partner(df_bs, df_p_bs)
    if plot == True:
        # print(np.max(df_part_mp_in))
        bins = 120
        plt.hist(df_part_mp.startDif, bins, range=[
                 0, 3610], alpha=0.5, label='MP', color='k')
        plt.hist(df_part_bs.startDif, bins, range=[
                 0, 3610], alpha=0.5, label='BS', color='r', ls='--', histtype='step')

        plt.legend()
        # plt.ylim(0,100)
        plt.yscale('log')
        plt.xlim(0, 3700)
        plt.xlabel('Time between start (s)')

    return df_part_bs, df_part_mp


# Check where crossings have no overlap looking at partner list created above
def crossing_disagree(df, t, n=10):
    ''' Finds the time difference between partner crossings and 
    returns a data frame of the n largest.
    Inputs:
    df -- partner data frames from output of compare_lists()
    t -- the largest time difference between partners of interest in seconds
    n -- Number of largest differences interested in (default n=10)
    '''
    x = df.index[~(((df['start'] >= df['startP'])
                    & (df['start'] <= df['endP'])) | ((df['end'] >= df['startP'])
                                                      & (df['end'] <= df['endP'])) | ((df['startP'] >= df['start'])
                                                                                      & (df['startP'] <= df['end'])) | ((df['endP'] >= df['start'])
                                                                                                                        & (df['endP'] <= df['end'])))]
    # print(x)
    df_disagree = df[df.index.isin(x)]

    # Find largest disagreement as difference between average time?
    def AvgDateDif(df):
        avg_date = (df[['start', 'end']].mean(axis=1))
        avg_dateP = (df[['startP', 'endP']].mean(axis=1))
        df = df.assign(AvgDateDiff=(
            avg_date-avg_dateP).abs().dt.total_seconds())
        return df

    df = AvgDateDif(df_disagree)
    print(len(df))
    df = df.loc[df['AvgDateDiff'] < t]
    print(len(df))
    largest10 = df.nlargest(n, 'AvgDateDiff')

    return largest10[['start', 'end', 'startP', 'endP', 'AvgDateDiff']]


def orbits_without_bs(df):
    ''' Returns orbits with no bowshock crossings after mp_out
    Input
    df -- crossing list dataframe 
    '''
    prev = 'mp'
    j = 0
    x = []
    for i in df['Type']:
        if i == ('mp_out'):
            current = 'mp'
            if current == prev:
                x.append(j)
            prev = 'mp'
        elif i.startswith('bs'):
            prev = 'bs'
        j += 1
    print(len(x))
    df_no_bs = df[df.index.isin(x)]

    return df_no_bs


def crossing_duration(df, df_sun, seperate=False):
    '''Plotting histograms of magnetopause crossing intervals
    Inputs:
    df -- philpott crossings dataframe
    df_sun -- Sun crossings dataframe
    seperate -- when true seperates histograms into in and out bound crossings (boolean, Default = False)
    '''
    df['Interval'] = (df.end-df.start).dt.total_seconds()

    df_sun['Interval'] = (df_sun.end-df_sun.start).dt.total_seconds()

    bins = 100
    if seperate == True:
        df_mp_in = df[(df.Type == 'mp_in')]
        df_mp_out = df[(df.Type == 'mp_out')]
        df2_mp_in = df_sun[(df_sun.Type == 'mp_in')]
        df2_mp_out = df_sun[(df_sun.Type == 'mp_out')]

        plt.hist(df_mp_in.Interval, bins, alpha=0.5, 
                 label=f'p_mp_in: n = {len(df_mp_in)}', color='k', histtype='step')
        plt.hist(df_mp_out.Interval, bins, alpha=0.5,
                 label=f'p_mp_out: n = {len(df_mp_out)}', color='b')
        plt.hist(df2_mp_in.Interval, bins, alpha=0.5, 
                 label=f'sun_mp_in: n = {len(df2_mp_in)}', color='darkred', histtype='step')
        plt.hist(df2_mp_out.Interval, bins, alpha=0.5, 
                 label=f's_mp_out: n = {len(df2_mp_out)}', color='yellow')
        plt.legend()
        plt.yscale('log')
        # plt.xlim(0,3700)
        plt.xlabel('Duration of mp crossing (s)')
    else:
        df_mp = df[(df.Type == 'mp_in') | (df.Type == 'mp_out')]
        df_sun_mp = df_sun[(df_sun.Type == 'mp_in') |
                           (df_sun.Type == 'mp_out')]
        ymax = df_sun.Interval.max()
        plt.hist(df_mp.Interval, bins, alpha=0.5, 
                 label=f'p_mp: n = {len(df_mp)}', color='k', histtype='step')
        plt.hist(df_sun_mp.Interval, bins, alpha=0.5,
                 label=f'sun_mp: n = {len(df_sun_mp)}', color='red')
        mean_p = df_mp.Interval.mean()
        mean_sun = df_sun_mp.Interval.mean()
        plt.vlines(mean_p, ymin=0, ymax=ymax, linestyles='--',
                   colors='k', label=f'P Mean = {mean_p:.1f}')
        plt.vlines(mean_sun, ymin=0, ymax=ymax, linestyles='--',
                   colors='red', label=f'Sun Mean = {mean_sun:.1f}')
        plt.legend()
        plt.yscale('log')
        # plt.xlim(0,3700)
        plt.xlabel('Duration of mp crossing (s)')
        return df_mp, df_sun_mp


# Returns an array of number of crossings per orbit and any
# orbits with crossings !=4. Need to run df through orbits first
def orbit_crossings(df):
    '''Function that returns number of orbits without 4
    crossings and dataframe of those orbits
    Input:
    df -- dataframe
    '''
    NumCrossings = []
    counter = 0
    for i in df['Orbit'].unique():
        num=len(df.loc[df['Orbit'] == i])
        NumCrossings.append(num)
    NumCrossings = np.asarray(NumCrossings)
    # Print orbit number where there is not 4 crossings
    weird_crossings = np.where(NumCrossings != 4)
    print(len(weird_crossings[0]))
    return NumCrossings, weird_crossings

def time_in_sheath(df, df_sun):
    '''Histogram of duration of time in sheath for each crossing 
    Inputs:
    df -- Philpott crossing dataframe
    df_sun -- Sun crossing dataframe
    '''
    # mp_out(end)-bs_out(start)
    # bs_in(end)-mp_in(start)

    # Function to find duration of time in Sheath for inbound and outbound
    def sheath(df):
        Dur_out = []
        Dur_in = []
        for i in df.index[:-1]:
            if df.Type[i] == 'mp_out' and df.Type[i+1] == 'bs_out':
                Dur_out.append((df.start[i+1]-df.end[i]).total_seconds())
            elif df.Type[i] == 'bs_in' and df.Type[i+1] == 'mp_in':
                Dur_in.append((df.start[i+1]-df.end[i]).total_seconds())
        return Dur_in, Dur_out
    # Philpott crossings
    Dur_in_p, Dur_out_p = sheath(df)
    # Sun crossings
    Dur_in_s, Dur_out_s = sheath(df_sun)
    bins = 100
    plt.hist(Dur_in_p, bins=bins, alpha=0.5,
             label='Sheath_in_p', color='k', histtype='step')
    plt.hist(Dur_out_p, bins=bins, alpha=0.5,
             label='Sheath_out_p', color='b', histtype='step')
    plt.hist(Dur_in_s, bins=bins, alpha=0.5, label='Sheath_in_s',
             color='orange', histtype='step')
    plt.hist(Dur_out_s, bins=bins, alpha=0.5, label='Sheath_out_s',
             color='darkred', histtype='step')
    plt.legend()
    # plt.yscale('log')
    plt.xlabel('Duration of time in sheath (s)')
    return


def save_largest(df, sun=False, philpott=False):
    '''
    Save the timeseries for the 10 longest crossing durations
    Inputs
    df -- crossings dataframe
    sun -- If true plots crossing lines identified by Sun (boolean, default = False)
    philpott -- If true plots crossing lines identified by Philpott (boolean, default = False)
    '''
    df = df.nlargest(10, 'Interval')
    minute = datetime.timedelta(minutes=10)
    start_time = []
    end_time = []
    for i in df.start:
        i = i-minute
        parsed_datetime = datetime.datetime.strptime(
            str(i), "%Y-%m-%d %H:%M:%S")
        output_datetime_str = parsed_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        start_time.append(str(output_datetime_str))
    for i in df.end:
        i = i+minute
        parsed_datetime = datetime.datetime.strptime(
            str(i), "%Y-%m-%d %H:%M:%S")
        output_datetime_str = parsed_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        end_time.append(str(output_datetime_str))
    for i in range(0, 10):
        if sun == True:
            mag.mag_time_series(
                start_date=start_time[i], end_date=end_time[i], sun=True, save=True, num=i)
        if philpott == True:
            mag.mag_time_series(
                start_date=start_time[i], end_date=end_time[i], philpott=True, save=True, num=i)



# combining all 60 second resolution data into one .csv file to find orbits
def find_tab_files(root_dir):
    """Recursively find all .TAB files in root_dir and its subdirectories."""
    tab_files = []
    for path, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('60_V08.TAB'):
                tab_files.append(os.path.join(path, file))
    tab_files = np.sort(tab_files)
    return tab_files


def combine_tab_files(tab_files, output_file):
    """Combine multiple .TAB files into one CSV file using np.genfromtxt."""
    combined_data = []

    for file in tab_files:
        data = np.genfromtxt(file, skip_header=0)
        combined_data.append(data)

    # Concatenate all the data arrays
    combined_data = np.concatenate(combined_data)
    # Save to a new CSV file
    np.savetxt(output_file, combined_data, delimiter=',')


def read_in_all():
    '''Read in all.csv which is 10 minute resolution ephemeris data for MESSENGER
    '''
    from datetime import datetime, timedelta

    # Read the CSV file into a DataFrame
    df = pd.read_csv('all.csv', header=None)
    df.columns = ['year', 'day_of_year', 'hour', 'minute', 'second', 'col6', 'col7',
                  'eph_x', 'eph_y', 'eph_z', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16']
    
    # Combine the time components into a single datetime column
    def combine_time_components(row):
        year = int(row['year'])
        day_of_year = int(row['day_of_year'])
        hour = int(row['hour'])
        minute = int(row['minute'])
        second = float(row['second'])

        # Combine the time components into a datetime object
        date = datetime(year, 1, 1) + timedelta(days=day_of_year -
                                                1, hours=hour, minutes=minute, seconds=second)
        return date

    # Apply the function to each row to create the 'Time' column
    df['Time'] = df.apply(combine_time_components, axis=1)

    # Select the relevant columns
    df = df[['Time', 'eph_x', 'eph_y', 'eph_z']]
    date_string = df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    ephx = df['eph_x']
    ephy = df['eph_y']
    msm_ephx = []
    msm_ephy = []
    for i in tqdm.tqdm(range(0, len(date_string))):
        phi = load_messenger_mag.get_aberration_angle(date_string[i])
        new_ephx = ephx[i]*np.cos(phi)-ephy[i]*np.sin(phi)
        new_ephy = ephx[i]*np.sin(phi)+ephy[i]*np.cos(phi)
        msm_ephx.append(new_ephx)
        msm_ephy.append(new_ephy)

    df['eph_x'] = msm_ephx
    df['eph_y'] = msm_ephy
    # Display the resulting DataFrame
    return df


def orbit(df_all, df_crossing):
    ''' Adds orbit number to crossing list dataframes
    Inputs: 
    df_all -- dataframe containing ephemeris
            data at a 10 minute resolution obtained from "df_all=read_in_all()"
    df_crossing -- dataframe containing crossing data
    '''
    # Finding radial distance 
    r_all = np.sqrt(df_all.eph_x.to_numpy()**2 + df_all.eph_y.to_numpy()**2 +
                    df_all.eph_z.to_numpy()**2)

    # Finding the minima of radial distance and calling min-to-min 1 orbit 
    peaks = find_peaks(-r_all, distance=36)
    orbits = 1
    df_crossing['Orbit'] = 0
    for p in tqdm.tqdm(range(len(peaks[0])-1)):
        orbit_range = [peaks[0][p], peaks[0][p+1]]
        time_range = [df_all.Time.iloc[orbit_range[0]],
                      df_all.Time.iloc[orbit_range[1]]]
        df_crossing.loc[(df_crossing['start'] <= time_range[1]) & (
            df_crossing['start'] >= time_range[0]), 'Orbit'] = int(orbits)
        orbits += 1
    orbit_range = [peaks[0][-1]]
    time_range = [df_all.Time.iloc[orbit_range[0]]]
    df_p = pd.read_pickle("df_p.pickle")
    df_p.loc[df_p['start'] >= time_range[0], 'Orbit'] = int(orbits)

    return df_crossing


def sign_change_count(df):
    '''Function to count the number of sign changes in a df column, to count rotations
    Input
    df -- dataframe column such a df['mag_x'] 
    '''
    changes = 0
    prev = df.values[0]/abs(df.values[0])
    for i in df:
        if i < 0:
            sign = -1
        else:
            sign = 1
        if sign == -prev:
            changes += 1
            prev = sign
    return int(changes)


def analyse_mp_crossings_sheath(df, n=10, largest=True):
    ''' Function to add mean, standard deviation, and number of rotations in the
    magentosheath to the crossing dataframe
    Input:
    df -- crossing dataframe
    n -- number of crossings considered (int, default = 10)
    largest -- Looking at n largest crossings, if False looks at smallest (boolean, default = True) 
    '''
    df = df[(df.Type == 'mp_in') | (df.Type == 'mp_out')]
    minutes = datetime.timedelta(minutes=10)
    df['Interval'] = (df.end-df.start).dt.total_seconds()

    if largest == True:
        df = df.nlargest(n, 'Interval')
    else:
        df = df.nsmallest(n, 'Interval')

    def add_info(df, i, df_info, orient):
        sign_change = sign_change_count(df_info[f'mag_{orient}'])
        df.loc[i, f'rotations_{orient}_10'] = (sign_change)
        df.loc[i, f'avg_B{orient}_10'] = (df_info[f'mag_{orient}'].mean())
        df.loc[i, f'std_B{orient}_10'] = (df_info[f'mag_{orient}'].std())
        half = int(len(df_info)/2)
        if df.loc[i, 'Type'] == 'mp_in':
            df_info_5 = df_info.iloc[half:]
        else:
            df_info_5 = df_info.iloc[:-half]
        sign_change_5 = sign_change_count(df_info_5[f'mag_{orient}'])
        df.loc[i, f'rotations_{orient}_5'] = (sign_change_5)
        df.loc[i, f'avg_B{orient}_5'] = (df_info_5[f'mag_{orient}'].mean())
        df.loc[i, f'std_B{orient}_5'] = (df_info_5[f'mag_{orient}'].std())
        if orient == 'z':
            df.loc[i, f'avg_BAmp_10'] = (df_info[f'magamp'].mean())
            df.loc[i, f'avg_BAmp_5'] = (df_info_5[f'magamp'].mean())
        return df

    for i in df.index:
        if df.loc[i, 'Type'] == 'mp_in':
            parsed_datetime_s = datetime.datetime.strptime(
                str(df.loc[i, 'start']-minutes), "%Y-%m-%d %H:%M:%S")
            start_str = parsed_datetime_s.strftime("%Y-%m-%d-%H-%M-%S")
            parsed_datetime_e = datetime.datetime.strptime(
                str(df.loc[i, 'start']), "%Y-%m-%d %H:%M:%S")
            end_str = parsed_datetime_e.strftime("%Y-%m-%d-%H-%M-%S")
            df_info = mag.mag_time_series(
                start_date=start_str, end_date=end_str, plot=False)
            # Add to new column
            df = add_info(df, i, df_info, 'x')
            df = add_info(df, i, df_info, 'y')
            df = add_info(df, i, df_info, 'z')

        if df.loc[i, 'Type'] == 'mp_out':
            parsed_datetime_s = datetime.datetime.strptime(
                str(df.loc[i, 'end']), "%Y-%m-%d %H:%M:%S")
            start_str = parsed_datetime_s.strftime("%Y-%m-%d-%H-%M-%S")
            parsed_datetime_e = datetime.datetime.strptime(
                str(df.loc[i, 'end']+minutes), "%Y-%m-%d %H:%M:%S")
            end_str = parsed_datetime_e.strftime("%Y-%m-%d-%H-%M-%S")
            df_info = mag.mag_time_series(
                start_date=start_str, end_date=end_str, plot=False)
            # Add to new column
            df = add_info(df, i, df_info, 'x')
            df = add_info(df, i, df_info, 'y')
            df = add_info(df, i, df_info, 'z')

    return df


# Histograms of Bx in MP, MSheath, and MSphere
def mag_mp_hist(df, n=1, minute=5, time_win=120, combine=False, MS=False, timeseries=True, splitDistro=False, dayside=False, RMS='total'):
    '''Plot histograms of B_x in Magnetopause, Magnetosheath and Magnetosphere
    Fits a double Gaussian to identify crossing location, Magnetosheath and Magnetosphere
    during crossing interval
    Inputs:
    df -- Dataframe of crossings
    n -- Number of largest crossings to be considered (int, default = 1)
    minute -- Amount of minutes considered each side of crossing (int, default = 5)
    time_win -- time window, in seconds, used to average RMS (int, default = 120)
    combine -- plot all histograms on the one plot (boolean, default = False)
    MS -- Plot histograms of Magnetosheath and Magnetosphere (boolean, default = False)
    timeseries -- plot time series for crossing (boolean, default = True)
    splitDistro -- plot a histogram of the split location 
    dayside -- True plots only dayside crossings (boolean, default = False)
    RMS -- Plots the RMS of the selected orientation (string, default = 'total', options = 'x','y','z') 
    '''
    from scipy.optimize import curve_fit

    def Gauss(x, mu, Sigma, A):
        return A*np.exp(-(x-mu)**2/(2*Sigma**2))

    def DoubleGaussian(x, mu1, Sigma1, A1, mu2, Sigma2, A2):
        return Gauss(x, mu1, Sigma1, A1)+Gauss(x, mu2, Sigma2, A2)

    df = df[(df.Type == 'mp_in') | (df.Type == 'mp_out')]
    df['Interval'] = (df.end-df.start).dt.total_seconds()

    if dayside == True:
        df = df[(df['start_x_msm'] > 0) & (df['end_x_msm'] > 0)]
    df = df.nlargest(n, 'Interval')
    minutes = datetime.timedelta(minutes=minute)
    if combine == False:
        split_middle = []
        split_std = []
        for i in df.index:
            # if i != 11317:
            #     continue
            parsed_datetime_s = datetime.datetime.strptime(
                str(df.loc[i, 'start']), "%Y-%m-%d %H:%M:%S")
            start_str = parsed_datetime_s.strftime("%Y-%m-%d-%H-%M-%S")
            parsed_datetime_e = datetime.datetime.strptime(
                str(df.loc[i, 'end']), "%Y-%m-%d %H:%M:%S")
            end_str = parsed_datetime_e.strftime("%Y-%m-%d-%H-%M-%S")
            df_info = mag.mag_time_series(
                start_date=start_str, end_date=end_str, plot=False)
            # bins= 30
            bins = int(np.sqrt(len(df_info['mag_x'])))
            if bins <= 0:
                bins = 50

            x_fit = np.linspace(
                df_info['mag_x'].min(), df_info['mag_x'].max(), 1000)
            y, x, _ = plt.hist(df_info['mag_x'].values, density=True, bins=bins, label=f'Orbit = {
                               df.loc[i, 'Orbit']}')

            # Peaks must be 5 bins away from eachother to count
            dis = 5*((df_info['mag_x'].max() - df_info['mag_x'].min())/bins)
            prom = 0.01

            # Finding peaks that match params if 2 or more 
            if len(find_peaks(y, height=np.mean(y), prominence=prom, distance=dis)[0]) >= 2:
                print('BIMODAL')
                bimodal = True
                # Finding height of two tallest peaks
                A1, A2 = np.sort(find_peaks(y, height=np.mean(
                    y), prominence=prom, distance=dis)[1]['peak_heights'])[-2:]

                # Finding corrresponding x position of the two tallest peaks
                x1 = np.where(find_peaks(y, height=np.mean(y), prominence=prom, distance=dis)[
                              1]['peak_heights'] == A1)[0][0]
                x2 = np.where(find_peaks(y, height=np.mean(y), prominence=prom, distance=dis)[
                              1]['peak_heights'] == A2)[0][0]
                x1 = find_peaks(y, height=np.mean(
                    y), prominence=prom, distance=dis)[0][x1]
                x2 = find_peaks(y, height=np.mean(
                    y), prominence=prom, distance=dis)[0][x2]

                mu1 = x[x1]
                mu2 = x[x2]

                # Using these values as the expected values for fitting
                expected = (mu1, 5, A1, mu2, 5, A2)

                params, cov = curve_fit(DoubleGaussian, x[:-1], y[:], expected)
                sigma = np.sqrt(np.diag(cov))
                # print(expected)
                print(params)
                # Defining split as where the two gaussians join
                split = (
                    x_fit[find_peaks(-DoubleGaussian(x_fit, *params))[0][0]])
                split_middle.append(split)
                # Code to define split where std is equal from both peaks
                if params[0] < params[3]:
                    peak1 = params[0]
                    sigma1 = params[1]
                    peak2 = params[3]
                    sigma2 = params[4]
                else:
                    peak1 = params[3]
                    sigma1 = params[4]
                    peak2 = params[0]
                    sigma2 = params[1]
                n = (peak2-peak1)/(sigma1+sigma2)
                print(n)
                split1 = peak1+n*sigma1
                split_std.append(split1)
                split2 = peak2-n*sigma2
                plt.title('Magnetopause interval')
                plt.vlines(split1, ymin=0, ymax=y.max(), colors='r',
                           ls='--', label='Split at same std')

                plt.vlines(split, ymin=0, ymax=y.max(), colors='k',
                           ls='--', label='Split at Intersection')
                plt.xlim(-60, 60)
                plt.plot(x_fit, DoubleGaussian(x_fit, *params),
                         color='red', lw=3, label='Double Gaussian Fit')
                plt.ylabel('Number of Measurements')
                plt.xlabel('$B_x$ in Magnetopause')
                plt.legend()
                plt.savefig(f'Images/Distro_mp_{i}.png')
                plt.show()
            
            # Less strict parameter if 2 or more peaks were not already found
            elif len(find_peaks(y, height=0.005, prominence=0.001, distance=dis)[0]) >= 2:
                print('BIMODAL RELAXED')
                bimodal = True
                # 0.001
                prom = 0.001
                # 0.005
                h = 0.005
                # Finding height of two tallest peaks
                A1, A2 = np.sort(find_peaks(y, height=h, prominence=prom, distance=dis)[
                                 1]['peak_heights'])[-2:]

                # Finding corrresponding x position of the two tallest peaks
                x1 = np.where(find_peaks(y, height=h, prominence=prom, distance=dis)[
                              1]['peak_heights'] == A1)[0][0]
                x2 = np.where(find_peaks(y, height=h, prominence=prom, distance=dis)[
                              1]['peak_heights'] == A2)[0][0]
                x1 = find_peaks(y, height=h, prominence=prom,
                                distance=dis)[0][x1]
                x2 = find_peaks(y, height=h, prominence=prom,
                                distance=dis)[0][x2]

                mu1 = x[x1]
                mu2 = x[x2]

                # Using these values as the expected values for fitting
                expected = (mu1, 5, A1, mu2, 5, A2)

                params, cov = curve_fit(DoubleGaussian, x[:-1], y[:], expected)
                sigma = np.sqrt(np.diag(cov))
                # print(expected)
                print(params)
                print(i)
                # Defining split as where the two gaussians join
                if len(find_peaks(-DoubleGaussian(x_fit, *params))[0]) > 0:
                    split = (
                        x_fit[find_peaks(-DoubleGaussian(x_fit, *params))[0][0]])
                else:
                    split = np.nan
                split_middle.append(split)
                # Code to define split where equal std from both peaks
                if params[0] < params[3]:
                    peak1 = params[0]
                    sigma1 = params[1]
                    peak2 = params[3]
                    sigma2 = params[4]
                else:
                    peak1 = params[3]
                    sigma1 = params[4]
                    peak2 = params[0]
                    sigma2 = params[1]
                n = (peak2-peak1)/(sigma1+sigma2)
                print(n)
                split1 = peak1+n*sigma1
                split_std.append(split1)
                split2 = peak2-n*sigma2
                plt.title('Magnetopause interval')
                plt.vlines(split1, ymin=0, ymax=y.max(), colors='r',
                           ls='--', label='Split at same std')

                plt.vlines(split, ymin=0, ymax=y.max(), colors='k',
                           ls='--', label='Split at Intersection')
                plt.xlim(-60, 60)
                plt.plot(x_fit, DoubleGaussian(x_fit, *params),
                         color='red', lw=3, label='Double Gaussian Fit')
                plt.ylabel('Number of Measurements')
                plt.xlabel('$B_x$ in Magnetopause')
                plt.legend()
                plt.savefig(f'Images/Distro_mp_{i}.png')
                plt.show()
                bimodal = True

            # If two or more peaks were not found in either case, look for 1
            else:
                # bimodal = False
                bimodal = True

                A = np.sort(find_peaks(y, height=np.mean(y), prominence=prom, distance=dis)[
                            1]['peak_heights'])[-1:]

                # Finding corrresponding x position of the tallest peak
                x1 = np.where(find_peaks(y, height=np.mean(y), prominence=prom, distance=dis)[
                              1]['peak_heights'] == A)[0][0]
                x1 = find_peaks(y, height=np.mean(
                    y), prominence=prom, distance=dis)[0][x1]
                mu1 = x[x1]

                # Using these values as the expected values for fitting
                expected = (mu1, 5, A1)

                print('NO BIMODAL')
                params, cov = curve_fit(Gauss, x[:-1], y[:], expected)
                plt.plot(x_fit, Gauss(x_fit, *params),
                         color='red', lw=3, label='Gaussian Fit')

                split = params[0]
                split_middle.append(split)

                plt.title('Magnetopause interval')
                plt.hist(df_info['mag_x'].values, density=True,
                         bins=bins, label=f'Orbit = {df.loc[i, 'Orbit']}')
                plt.xlim(-60, 60)
                plt.vlines(split, ymin=0, ymax=y.max(), colors='k',
                           ls='--', label='Split at Peak')
                plt.ylabel('Number of Measurements')
                plt.xlabel('$B_x$ in Magnetopause')
                plt.legend()
                plt.savefig(f'Images/Distro_mp_{i}.png')
                plt.show()

            parsed_datetime_s = datetime.datetime.strptime(
                str(df.loc[i, 'start']-minutes), "%Y-%m-%d %H:%M:%S")
            start_str = parsed_datetime_s.strftime("%Y-%m-%d-%H-%M-%S")
            Total_start = start_str
            parsed_datetime_e = datetime.datetime.strptime(
                str(df.loc[i, 'start']), "%Y-%m-%d %H:%M:%S")
            end_str = parsed_datetime_e.strftime("%Y-%m-%d-%H-%M-%S")
            df_info = mag.mag_time_series(
                start_date=start_str, end_date=end_str, plot=False)
            bins = 10
            if MS == True:
                plt.title(f'{minute} minutes before MP')
                plt.hist(df_info['mag_x'].values, bins=bins,
                         label=f'Orbit = {df.loc[i, 'Orbit']}')
                plt.ylabel('Number of Measurements')
                if df.loc[i, 'Type'] == 'mp_in':
                    plt.title(f'Magnetosheath {minute} minutes before MP interval for inbound')
                    plt.xlabel('$B_x$ in magnetosheath')
                else:
                    plt.title(f'Magnetosphere {minute} minutes before MP interval for outbound')
                    plt.xlabel('$B_x$ in magnetosphere')
                plt.legend()
            # plt.savefig('Before_MP.png')
            # plt.show()

            parsed_datetime_s = datetime.datetime.strptime(
                str(df.loc[i, 'end']), "%Y-%m-%d %H:%M:%S")
            start_str = parsed_datetime_s.strftime("%Y-%m-%d-%H-%M-%S")
            parsed_datetime_e = datetime.datetime.strptime(
                str(df.loc[i, 'end']+minutes), "%Y-%m-%d %H:%M:%S")
            end_str = parsed_datetime_e.strftime("%Y-%m-%d-%H-%M-%S")
            Total_end = end_str
            df_info = mag.mag_time_series(
                start_date=start_str, end_date=end_str, plot=False)
            bins = 10
            if MS == True:
                plt.hist(df_info['mag_x'].values, bins=bins,
                         label=f'Orbit = {df.loc[i, 'Orbit']}')
                plt.ylabel('Number of Measurements')
                if df.loc[i, 'Type'] == 'mp_in':
                    plt.title(f'Magnetosphere {minute} minutes after MP interval for inbound')
                    plt.xlabel('$B_x$ in magnetosheath')
                else:
                    plt.title(f'Magnetosheath {minute} minutes after MP interval for outbound')
                    plt.xlabel('$B_x$ in magnetosheath')
                plt.legend()
                # plt.savefig('After_MP.png')
                # plt.show()

            df_info = mag.mag_time_series(
                start_date=Total_start, end_date=Total_end, plot=False)
            # plotting time series with split colour coded
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(
                4, 1, sharex=True, figsize=(12, 8), dpi=300)
            Bamp = df_info['magamp']
            x = df_info['Time']
            y = df_info['mag_x']
            # Colour coding magnetosheath, magnetosphere and magnetopause
            if df.loc[i, 'Type'] == 'mp_in':
                Sheath = np.ma.masked_where(x > df.loc[i, 'start'], x)
                Sphere = np.ma.masked_where(x < df.loc[i, 'end'], x)
                SheathAmp = np.ma.masked_where(x > df.loc[i, 'start'], Bamp)
                SphereAmp = np.ma.masked_where(x < df.loc[i, 'end'], Bamp)
                Pause = np.ma.masked_where(
                    (x > df.loc[i, 'end']) | (x < df.loc[i, 'start']), x)
            else:
                Sphere = np.ma.masked_where(x > df.loc[i, 'start'], x)
                Sheath = np.ma.masked_where(x < df.loc[i, 'end'], x)
                SphereAmp = np.ma.masked_where(x > df.loc[i, 'start'], Bamp)
                SheathAmp = np.ma.masked_where(x < df.loc[i, 'end'], Bamp)
                Pause = np.ma.masked_where(
                    (x > df.loc[i, 'end']) | (x < df.loc[i, 'start']), x)

            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.plot(Sheath, y, c='b', label='Magnetosheath')
            ax1.plot(Sphere, y, c='red', label='Magnetosphere')
            ax1.plot(Pause, y, c='mediumpurple', label='Magnetopause')
            ax1.set_ylabel('$B_x$ (nT)')
            ax1.axhline(split, ls='--', c='r', lw=1)
            ax1.axhline(ls='--', c='k')
            ax1.axvline(df.loc[i, 'start'], c='k')
            ax1.axvline(df.loc[i, 'end'], c='k')
            ax1.legend()

            if bimodal == True:
                MagUpper = np.ma.masked_where(y < split, y)
                MagLower = np.ma.masked_where(y > split, y)

                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax2.plot(x, MagUpper, c='b', label='')
                ax2.plot(x, MagLower, c='r', label='')
                ax2.plot(Sheath, y, c='b', label='Magnetosheath')
                ax2.plot(Sphere, y, c='red', label='Magnetosphere')
                # ax2.set_xlabel('Time')
                ax2.set_ylabel('$B_x$ (nT)')
                ax2.axhline(split, ls='--', c='r', lw=1)
                ax2.axhline(ls='--', c='k')
                ax2.axvline(df.loc[i, 'start'], c='k')
                ax2.axvline(df.loc[i, 'end'], c='k')

                Bamp = df_info['magamp']
                BampUpper = np.ma.masked_where(y < split, Bamp)
                BampLower = np.ma.masked_where(y > split, Bamp)

                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax3.plot(x, BampUpper, c='b', label='')
                ax3.plot(x, BampLower, c='r', label='')
                ax3.plot(x, SheathAmp, c='b', label='')
                ax3.plot(x, SphereAmp, c='r', label='')
                # ax3.set_xlabel('Time')
                ax3.set_ylabel('$|B|$ (nT)')
                ax3.axhline(ls='--', c='k')
                ax3.axvline(df.loc[i, 'start'], c='k')
                ax3.axvline(df.loc[i, 'end'], c='k')

            else:
                ax2.plot(x, y)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax2.set_xlabel('Time')
                ax2.set_ylabel('$B_x$ (nT)')
                ax2.axhline(ls='--', c='k')
                ax2.axvline(df.loc[i, 'start'], c='k')
                ax2.axvline(df.loc[i, 'end'], c='k')

                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax3.plot(x, y, label='')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('$|B|$ (nT)')
                ax3.axhline(ls='--', c='k')
                ax3.axvline(df.loc[i, 'start'], c='k')
                ax3.axvline(df.loc[i, 'end'], c='k')

            # Add fourth panel with RMS
            time_int = int(time_win/2)
            if RMS == 'x':
                Bamp = y
                ax4.set_ylabel(f'RMS of $B_x$ \n averaging over {time_win} seconds')
            else:
                ax4.set_ylabel(f'RMS of $|B|$ \n averaging over {time_win} seconds')
            time_avg = x[time_int:-time_int]
            mag_avg = Bamp[time_int:-time_int]
            b_avg = np.array([])
            if np.size(x) > int(time_int*2):
                for p in np.round(np.linspace(time_int, np.size(x)-time_int, num=np.size(x)-int(time_int*2))):
                    gd = [p-time_int, p+time_int]
                    ba = Bamp[int(gd[0]):int(gd[1])]
                    b_avg = np.append(b_avg, np.mean(ba))
                mag_test = Bamp[time_int:-time_int]
                rms_B = np.sqrt((mag_test-b_avg)**2)

            date = df.loc[i, 'start']
            date = date.strftime("%Y-%m-%d")
            ax4.set_xlabel(f'Time \n Date: {date}')

            if df.loc[i, 'Type'] == 'mp_in':
                SheathRMS = np.ma.masked_where(
                    x[time_int:-time_int] > df.loc[i, 'start'], rms_B)
                SphereRMS = np.ma.masked_where(
                    x[time_int:-time_int] < df.loc[i, 'end'], rms_B)

            else:
                SphereRMS = np.ma.masked_where(
                    x[time_int:-time_int] > df.loc[i, 'start'], rms_B)
                SheathRMS = np.ma.masked_where(
                    x[time_int:-time_int] < df.loc[i, 'end'], rms_B)

            RMSUpper = np.ma.masked_where(y[time_int:-time_int] < split, rms_B)
            RMSLower = np.ma.masked_where(y[time_int:-time_int] > split, rms_B)

            # ax4.plot(time_avg,rms_B, c='gray',lw='0.75')
            ax4.plot(time_avg, RMSUpper, c='b', label='')
            ax4.plot(time_avg, RMSLower, c='r', label='')
            ax4.plot(time_avg, SheathRMS, c='b', label='')
            ax4.plot(time_avg, SphereRMS, c='r', label='')

            ax4.axvline(df.loc[i, 'start'], c='k')
            ax4.axvline(df.loc[i, 'end'], c='k')

            fig.savefig(f'Images/colour_coded_mp_{i}.png')
            fig.show()

            if timeseries == True:
                # print(Total_start,Total_end)
                axt, figt = mag.mag_time_series(
                    start_date=Total_start, end_date=Total_end, philpott=True, save=True, num=i)
                figt.savefig(f'Images/time_series_{i}.png')

        if splitDistro == True:
            plt.close()
            bins = 20
            plt.hist(split_middle, color='k', bins=bins,
                     label='Intersection Split', histtype='step')
            plt.hist(split_std, color='r', bins=bins,
                     label='Std Split', histtype='step')
            plt.title('Distribution of Splits')
            plt.legend()
            plt.xlabel('$B_x$ in Magnetopause')
            plt.ylabel('Measurements')

            plt.show()
    else:
        MP = []
        MSheath_in = []
        MSheath_out = []
        MSphere_in = []
        MSphere_out = []
        for i in df.index:
            parsed_datetime_s = datetime.datetime.strptime(
                str(df.loc[i, 'start']), "%Y-%m-%d %H:%M:%S")
            start_str = parsed_datetime_s.strftime("%Y-%m-%d-%H-%M-%S")
            parsed_datetime_e = datetime.datetime.strptime(
                str(df.loc[i, 'end']), "%Y-%m-%d %H:%M:%S")
            end_str = parsed_datetime_e.strftime("%Y-%m-%d-%H-%M-%S")
            df_info = mag.mag_time_series(
                start_date=start_str, end_date=end_str, plot=False)
            MP.append(df_info['mag_x'].values)

            parsed_datetime_s = datetime.datetime.strptime(
                str(df.loc[i, 'start']-minutes), "%Y-%m-%d %H:%M:%S")
            start_str = parsed_datetime_s.strftime("%Y-%m-%d-%H-%M-%S")
            parsed_datetime_e = datetime.datetime.strptime(
                str(df.loc[i, 'start']), "%Y-%m-%d %H:%M:%S")
            end_str = parsed_datetime_e.strftime("%Y-%m-%d-%H-%M-%S")
            df_info = mag.mag_time_series(
                start_date=start_str, end_date=end_str, plot=False)
            if df.loc[i, 'Type'] == 'mp_in':
                MSheath_in.append(df_info['mag_x'].values)
            else:
                MSphere_out.append(df_info['mag_x'].values)

            parsed_datetime_s = datetime.datetime.strptime(
                str(df.loc[i, 'end']), "%Y-%m-%d %H:%M:%S")
            start_str = parsed_datetime_s.strftime("%Y-%m-%d-%H-%M-%S")
            parsed_datetime_e = datetime.datetime.strptime(
                str(df.loc[i, 'end']+minutes), "%Y-%m-%d %H:%M:%S")
            end_str = parsed_datetime_e.strftime("%Y-%m-%d-%H-%M-%S")
            df_info = mag.mag_time_series(
                start_date=start_str, end_date=end_str, plot=False)
            if df.loc[i, 'Type'] == 'mp_in':
                MSphere_in.append(df_info['mag_x'].values)
            else:
                MSheath_out.append(df_info['mag_x'].values)
        bins = 50
        plt.title('Magnetopause interval')
        plt.hist(MP, bins=bins, density=True, label=f'Orbits = {n}')
        plt.ylabel('Number of Measurements')
        plt.xlabel('$B_x$ in Magnetopause')
        plt.legend()
        plt.savefig('In_MP.png')
        plt.show()

        # print(df_info)


def spatial_box_plot(df_crossing, orient='x'):
    ''' Plot box plots of spatial position versus crossing duration
    Crossings duration split into 4 equally sized bins
    Inputs:
    df_crossing -- crossings data frame  
    orient -- spatial dimension being plotted (string, default = 'x', options = 'y','z','all')
    '''
    df = df_crossing
    df['Interval'] = (df.end-df.start).dt.total_seconds()

    quant25 = df['Interval'].quantile(0.25)
    quant50 = df['Interval'].quantile(0.5)
    quant75 = df['Interval'].quantile(0.75)

    if orient == 'all':
        df['Avg_x'] = (df['start_x_msm']+df['end_x_msm'])/2
        df['Avg_y'] = (df['start_y_msm']+df['end_y_msm'])/2
        df['Avg_z'] = (df['start_z_msm']+df['end_z_msm'])/2
        r_all = np.sqrt(df.Avg_x.to_numpy()**2 + df.Avg_y.to_numpy()**2 +
                        df.Avg_z.to_numpy()**2)
        df['All'] = r_all
        bin1 = df['All'][df['Interval'] <= quant25].values
        bin2 = df['All'][(df['Interval'] > quant25) & (
            (df['Interval'] <= quant50))].values
        bin3 = df['All'][(df['Interval'] > quant50) & (
            (df['Interval'] <= quant75))].values
        bin4 = df['All'][(df['Interval'] > quant75)].values
        plt.ylabel('Position in $\sqrt{x^2+y^2+z^2}$_MSM ($R_M$)')

    else:
        df[f'Avg_{orient}'] = (
            df[f'start_{orient}_msm']+df[f'end_{orient}_msm'])/2
        bin1 = df[f'Avg_{orient}'][df['Interval'] <= quant25].values
        bin2 = df[f'Avg_{orient}'][(df['Interval'] > quant25) & (
            (df['Interval'] <= quant50))].values
        bin3 = df[f'Avg_{orient}'][(df['Interval'] > quant50) & (
            (df['Interval'] <= quant75))].values
        bin4 = df[f'Avg_{orient}'][(df['Interval'] > quant75)].values
        plt.ylabel(f'Position in {orient}_MSM ($R_M$)')

    labels = [f'0 - {quant25} s \n $n$={len(bin1)}', f' {quant25} - {quant50} s \n $n$={len(
        bin2)}', f'{quant50} - {quant75} s \n $n$={len(bin3)}', f'> {quant75} s \n $n$={len(bin4)}']
    total = [bin1, bin2, bin3, bin4]

    plt.boxplot(total, labels=labels, whis=(0, 100))
    # plt.violinplot(total)
    # plt.xticks([y + 1 for y in range(len(total))],
    #               labels=labels)
    print(len(bin1), len(bin2), len(bin3), len(bin4))
    plt.ylim(-8, 8)
    plt.xlabel('Duration of Magnetopause Crossing')
    plt.savefig(f'box_plot_{orient}.png')
    # plt.savefig(f'Violin_plot_{orient}.png')
    plt.show()


def add_type_to_all(df_all, df_crossing):
    ''' Adds type inforation to df_all, i.e. if in magnetosheath, or magnetoshere
    Inputs:
    df_all -- data frame containing containting time of mission
    df_crossing -- crossing list data frame 
    '''
    counter = 0
    df_all['Type'] = 'None'
    for i in tqdm.tqdm(df_crossing['Orbit'].unique()):
        start_values_out = df_crossing[(df_crossing['Orbit'] == i) & (
            df_crossing['Type'] == 'mp_out')]['end'].values
        end_values_out = df_crossing[(df_crossing['Orbit'] == i) & (
            df_crossing['Type'] == 'bs_out')]['start'].values

        start_values_in = df_crossing[(df_crossing['Orbit'] == i) & (
            df_crossing['Type'] == 'bs_in')]['end'].values
        end_values_in = df_crossing[(df_crossing['Orbit'] == i) & (
            df_crossing['Type'] == 'mp_in')]['start'].values

        start_sphere = df_crossing[(df_crossing['Orbit'] == i) & (
            df_crossing['Type'] == 'mp_in')]['end'].values
        hour = datetime.timedelta(hours=1)
        end_sphere = (df_crossing[(df_crossing['Orbit'] == i) & (
            df_crossing['Type'] == 'mp_out')]['start']).values

        # Check if there are any values found for start and end
        if start_values_out.size > 0 and end_values_out.size > 0:
            start = start_values_out[0]
            end = end_values_out[0]
            # Update the 'Type' column in df_all if its with Sheath time range
            df_all.loc[(df_all['Time'] > start) & (
                df_all['Time'] < end), 'Type'] = 'Sheath'

        if start_values_in.size > 0 and end_values_in.size > 0:
            start = start_values_in[0]
            end = end_values_in[0]
            # Update the 'Type' column in df_all if its with Sheath time range
            df_all.loc[(df_all['Time'] > start) & (
                df_all['Time'] < end), 'Type'] = 'Sheath'

        if start_sphere.size > 0 and end_sphere.size > 0:
            start = start_sphere[0]
            end = end_sphere[0]
            counter += 1
            # Update the 'Type' column in df_all if its with Sheath time range
            df_all.loc[(df_all['Time'] > start) & (
                df_all['Time'] < end), 'Type'] = 'Sphere'

    print(f"\n Orbits with sphere {counter}/{len(df_crossing['Orbit'].unique())}")
    return df_all

'''
# Code to read in df_info data frame if previous saved as pickle file
df_info = pd.read_pickle("df_info_all.pkl")
'''



def spatial_binning_orbits(df_info, df_crossing, add_type=False, plot_time_hist=False):
    ''' Function to bin magntosheath and magnetosphere into 4 areas x-z and y-z orbits
    split into positive and negative x and y. Then plot histograms of mag distributions
    Inputs:
    df_info -- data frame containing ephemeris and mag data at 1 second resolution for entire mission
    df_crossing -- crossing list data frame 
    add_type -- adds when in magnetosheath and magnetosphere to df_info, else loads in pickle with info (boolean, default = False)
    plot_time_hist -- plots histogram distribution of sheath and sphere over entire mission (boolean, default = False)
    '''

    if add_type == True:
        df_info = add_type_to_all(df_info, df_crossing)
    else:
        df_info = pd.read_pickle("df_info_type.pickle")

    df_all_sheath = df_info[(df_info.Type == 'Sheath')]
    df_all_sphere = df_info[(df_info.Type == 'Sphere')]

    df_all_sheath['Theta'] = np.rad2deg(
        np.arctan(df_all_sheath['eph_x']/df_all_sheath['eph_y']))
    df_all_sphere['Theta'] = np.rad2deg(
        np.arctan(df_all_sphere['eph_x']/df_all_sphere['eph_y']))

    def binning(df_all_sheath):
        xz1 = df_all_sheath[((df_all_sheath.Theta >= 45) | (
            df_all_sheath.Theta < -45)) & (df_all_sheath.eph_x <= 0)]
        yz1 = df_all_sheath[(df_all_sheath.Theta < 45) & (
            df_all_sheath.Theta >= -45) & (df_all_sheath.eph_y >= 0)]
        xz2 = df_all_sheath[((df_all_sheath.Theta >= 45) | (
            df_all_sheath.Theta < -45)) & (df_all_sheath.eph_x > 0)]
        yz2 = df_all_sheath[(df_all_sheath.Theta < 45) & (
            df_all_sheath.Theta >= -45) & (df_all_sheath.eph_y < 0)]
        return xz1, yz1, xz2, yz2

    label = ['x-z orbit \n & x <= 0', 'y-z orbit \n & y > 0',
             'x-z orbit \n & x > 0', 'y-z orbit \n & y <= 0']

    if plot_time_hist == True:
        bin1, bin2, bin3, bin4 = binning(df_all_sphere)
        counts = [len(bin1), len(bin2), len(bin3), len(bin4)]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.bar(label, counts)
        ax1.set_title('Sphere')
        ax1.set_ylabel('Time spent in region \n (1 minutes)')
        bin1, bin2, bin3, bin4 = binning(df_all_sheath)
        counts = [len(bin1), len(bin2), len(bin3), len(bin4)]
        ax2.bar(label, counts)
        ax2.set_title('Sheath')
        ax2.set_ylabel('Time spent in region \n (1 minutes)')
        fig.show()
        fig.savefig("SheathSphere.png")
        plt.show()

    xz1_sheath, yz1_sheath, xz2_sheath, yz2_sheath = binning(df_all_sheath)
    xz1_sphere, yz1_sphere, xz2_sphere, yz2_sphere = binning(df_all_sphere)

    '''
    # Code to save all ephemeris, mag, and time data at 1 second resolution to a pickle file
    
    parsed_datetime_s = datetime.datetime.strptime(str(df_all['Time'].min()), "%Y-%m-%d %H:%M:%S")
    start_str = parsed_datetime_s.strftime("%Y-%m-%d-%H-%M-%S")
    parsed_datetime_e = datetime.datetime.strptime(str(df_all['Time'].max()), "%Y-%m-%d %H:%M:%S")
    end_str = parsed_datetime_e.strftime("%Y-%m-%d-%H-%M-%S")
    print(start_str,end_str)

    df_info = mag.mag_time_series(start_date=start_str,end_date=end_str,plot=False)
    df_info.to_pickle("df_info_all.pkl")
    '''


    def hist(bin, title):

        var_name = title

        plt.title(f'{var_name} (Total counts={len(bin)})')
        plt.ylabel('Counts')
        plt.xlabel('Magnetic field strength (nT)')
        a = -100
        b = 100
        plt.hist(bin['mag_x'], bins=100, range=(
            a, b), histtype='step', label='$B_x$')
        plt.hist(bin['mag_y'], bins=100, range=(
            a, b), histtype='step', label='$B_y$')
        plt.hist(bin['mag_z'], bins=100, range=(
            a, b), histtype='step', label='$B_z$')
        plt.hist(bin['magamp'], bins=100, range=(
            a, b), histtype='step', label='|B|')
        plt.legend()
        # plt.xlim(-100,100)
        # plt.yscale('log')
        plt.savefig(f'{var_name}.png')
        plt.show()

        fig, ax = plt.subplots()

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        df = pd.DataFrame([[round(bin['mag_x'].min(),2),round(bin['mag_x'].max(),2) , round(bin['mag_x'].median(),2)], [round(bin['mag_y'].min(),2),round(bin['mag_y'].max(),2) , round(bin['mag_y'].median(),2)], [round(bin['mag_z'].min(),2),round(bin['mag_z'].max(),2) , round(bin['mag_z'].median(),2)],[round(bin['magamp'].min(),2),round(bin['magamp'].max(),2) , round(bin['magamp'].median(),2)]], index=['B_x','B_y','B_z','|B|'],columns=['min (nT)','max (nT)','median (nT)'])

        ax.table(cellText=df.values, colLabels=df.columns,rowLabels=df.index,loc='center')

        fig.tight_layout()
        fig.savefig(f'Table_{var_name}.png')
        plt.show()

    hist(xz1_sheath, 'xz1_sheath')
    hist(yz1_sheath, 'yz1_sheath')
    hist(xz2_sheath, 'xz2_sheath')
    hist(yz2_sheath, 'yz2_sheath')
    hist(xz1_sphere, 'xz1_sphere')
    hist(yz1_sphere, 'yz1_sphere')
    hist(xz2_sphere, 'xz2_sphere')
    hist(yz2_sphere, 'yz2_sphere')

    # plt.hist(xz1_sheath['mag_x'])

#!!!Plot may be flipped, need check if dawn side is switched with day side !!! 
def residence_plot(df_info, first_time=False):
    ''' Polar residence plot of percentage time in sphere compared to Sheath around Mercury
    Inputs:
    df_info -- dataframe containing ephemeris data
    first_time -- When true adds sheath and sphere label to df_info, must be true for first run
    '''

    # df_crossing = df_p
    # df_info=add_type_to_all(df_info,df_crossing)
    # df_info.to_pickle("df_info_type.pickle")

    df_info = pd.read_pickle("df_info_type.pickle")
    df = df_info.loc[(df_info['Type'] == 'Sheath') | (df_info['Type']=='Sphere')]

    # Calculate angles

    df['x'] = df['eph_x']/2440
    df['y'] = df['eph_y']/2440
    df['r'] = np.sqrt(df['y']**2+df['x']**2)
    df['angle'] = np.arctan2(df['y'], df['x'])

    # Ensure angles are in the range [0, 2*pi]
    df['angle'] = np.where(
        df['angle'] < 0, df['angle'] + 2 * np.pi, df['angle'])

    # Number of bins
    num_bins_angle = 36
    num_bins_radius = 20

    # Binning the data
    bins_angle = np.linspace(0, 2 * np.pi, num_bins_angle + 1)
    bins_radius = np.linspace(0, 5, num_bins_radius + 1)
    # bins_radius = np.linspace(df['r'].min(), df['r'].max(), num_bins_radius + 1)

    df['angle_bin'] = pd.cut(df['angle'], bins=bins_angle,
                             labels=False, include_lowest=True)
    df['radius_bin'] = pd.cut(
        df['r'], bins=bins_radius, labels=False, include_lowest=True)

    bin_counts = np.zeros((num_bins_radius, num_bins_angle))
    sheath_counts = np.zeros((num_bins_radius, num_bins_angle))
    sphere_counts = np.zeros((num_bins_radius, num_bins_angle))

    if first_time == True:
        for _, row in df.iterrows():
            r_bin = int(row['radius_bin'])
            a_bin = int(row['angle_bin'])
            bin_counts[r_bin, a_bin] += 1
            if row['Type'] == 'Sheath':
                sheath_counts[r_bin, a_bin] += 1
            elif row['Type'] == 'Sphere':
                sphere_counts[r_bin, a_bin] += 1
        print(sphere_counts)
        print(sheath_counts)
        np.save('Bin_counts.npy', bin_counts)
        np.save('Sphere_counts.npy', sphere_counts)
        np.save('Sheath_counts.npy', sheath_counts)
    else:
        bin_counts = np.load('Bin_counts.npy')
        sphere_counts = np.load('Sphere_counts.npy')
        sheath_counts = np.load('Sheath_counts.npy')

    print('Sheath', np.shape(sheath_counts))
    print('Sphere', np.shape(sphere_counts))
    print('Bins', np.shape(bin_counts))
    # Calculate percentage spheres of bin sphere/total
    percent_sphere = np.zeros_like(bin_counts, dtype=float)
    nonzero_mask = bin_counts > 0
    percent_sphere[nonzero_mask] = sphere_counts[nonzero_mask]/bin_counts[nonzero_mask]
    percent_sphere[~nonzero_mask] = np.nan  # Assign NaN to bins with no data

    # Mask the bins with no data
    masked_percent_sphere = np.ma.masked_where(
        np.isnan(percent_sphere), percent_sphere)

    # Create a colormap with white for NaNs
    cmap = plt.get_cmap('coolwarm')
    cmap.set_bad(color='white')

    # Create the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    r, theta = np.meshgrid(bins_radius, bins_angle)

    # Plot data using pcolormesh
    c = ax.pcolor(theta, r, masked_percent_sphere.T,
                      cmap=cmap, shading='flat')  # , edgecolors='k')

    cbar = fig.colorbar(c, ax=ax, orientation='vertical') 
    cbar.set_ticks(ticks=[ 0, 0.5,1], labels=['100% Sheath', '50%', '100% Sphere'])

    # Add title
    ax.set_title('Spatial distrabution of magnetosphere compared to magnetosheath')

    ax.set_xticks(np.linspace(0, 4 * np.pi/2, 13))
    ax.set_xticklabels(['12', '14', '16', '18', '20', '22',
                       '0', '2', '4', '6', '8', '10', '12'])

    ax.set_yticks(np.linspace(1, 5.0, 5))
    ax.set_yticklabels(['1 $R_M$', '2 $R_M$', '3 $R_M$', '4 $R_M$', '5 $R_M$'])
    fig.savefig('PolarSpatial.png')
    plt.show()
