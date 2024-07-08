#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:33:16 2024

@author: bowersch
"""
''' Code to load in MESSENGER boundaries identified by Philpott and Sun.

 Link to download Philpott boundary list:  
    
     https://doi.org/10.1029/2019JA027544
    
   Want the jgra55678-sup-0002-Table_SI-S01.xlsx file in the 
   Supplementary Material section. 

   Then, save this file as a .csv file for more easy use in python


 Specify location on you machine of this file here:
    '''


'''
Load list if already saved in pickle
'''

import matplotlib.pyplot as plt
import ephem
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import load_messenger_mag as load_messenger_mag
import MESSENGER_Boundary_ID as mag
import tqdm

# with open('df_p.pickle', 'rb') as f:
#     df_p = pickle.load(f)
# with open('df_s.p', 'rb') as f:
#     df_sun = pickle.load(f)

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

# Load in packages


def convert_to_datetime(date_string):
    ''' converts date_string to datetime object'''
    import datetime
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

            import numpy as np

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
                import datetime
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
        import numpy as np
        import pandas as pd

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
    # number of bins
    numBin = 4
    MPmin = df_mp["Distance"].min()
    MPmax = df_mp["Distance"].max()
    MPBins = (MPmax-MPmin)/numBin

    # Create new dataframes based on distances
    min = MPmin
    Bins = MPBins

    '''
    #Working on having the number of bins be an input into the function
    df_mpN = []
    for i in range(0,numBin):
    '''

    df_mp1 = df_mp[df_mp["Distance"] < min+Bins]
    df_mp2 = df_mp[(df_mp["Distance"] > min+Bins) &
                   (df_mp["Distance"] < min+2*Bins)]
    df_mp3 = df_mp[(df_mp["Distance"] > min+2*Bins) &
                   (df_mp["Distance"] < min+3*Bins)]
    df_mp4 = df_mp[(df_mp["Distance"] > min+3*Bins) &
                   (df_mp["Distance"] < min+4*Bins)]

    # Dividing BS into 4 distance regions
    # number of bins
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


def plot_boundary_over_year(df, EndYrs, StartYrs=0):

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
    Function to plot the measured magnetopause distance from the nominal magnetopause obtained by Winslow et al., 2013  
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

    # finding the distance drom the nominal
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

    # Now plot mindist versus time Look @ slack to see how to plot datetime
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
    Plot timeseries of B fields in 3-axis with 2015 mag data
    WIP: to add ability to select start and end time
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


# Breakinf lists into 2 dif types, identifying difference betweem starts
# times and ploting histogram
def compare_lists(df, df_p, plot=False):

    # df = df[df.index > 14800]
    # df_p = df_p[df_p.index < 1600]

    def split(df):
        df_mp = df[(df.Type == 'mp_in') | (df.Type == 'mp_out')]
        df_bs = df[(df.Type == 'bs_in') | (df.Type == 'bs_out')]
        return df_mp, df_bs

    df_mp, df_bs = split(df)
    df_p_mp, df_p_bs = split(df_p)

    def find_partner(df, df_p):
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

        # Set 'Flag' to 2 where 'startDif' is greater than or equal to 3600
        df.loc[df['startDif'] >= 3600, 'Flag'] = 2
        extreme_count = df.loc[df.Flag == 2, 'Flag'].count()
        print("EXTREME:", extreme_count)

        # Set 'Flag' to 3 when orbit disagrees by just a few minutes (10 minutes)
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
def crossing_disagree(df, t):
    # df is partnered list, t is max time diff between crossing pair in seconds
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
    largest10 = df.nlargest(10, 'AvgDateDiff')

    return largest10[['start', 'end', 'startP', 'endP', 'AvgDateDiff']]


def orbits_without_bs(df):
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

# Plotting histograms of MP crossing intervals, Seperate = True seperates in and out
def crossing_duration(df, df_sun, seperate=False):

    df['Interval'] = (df.end-df.start).dt.total_seconds()

    df_sun['Interval'] = (df_sun.end-df_sun.start).dt.total_seconds()

    bins = 100
    if seperate == True:
        df_mp_in = df[(df.Type == 'mp_in')]
        df_mp_out = df[(df.Type == 'mp_out')]
        df2_mp_in = df_sun[(df_sun.Type == 'mp_in')]
        df2_mp_out = df_sun[(df_sun.Type == 'mp_out')]

        plt.hist(df_mp_in.Interval, bins, alpha=0.5, label=f'p_mp_in: n = {
                 len(df_mp_in)}', color='k', histtype='step')
        plt.hist(df_mp_out.Interval, bins, alpha=0.5,
                 label=f'p_mp_out: n = {len(df_mp_out)}', color='b')
        plt.hist(df2_mp_in.Interval, bins, alpha=0.5, label=f'sun_mp_in: n = {
                 len(df2_mp_in)}', color='darkred', histtype='step')
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
        plt.hist(df_mp.Interval, bins, alpha=0.5, label=f'p_mp: n = {len(df_mp)}', color='k', histtype='step')
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

#!!! Not fully working, come back to this!!!

def orbit_crossings(df):
    # weird_orbit = []
    # j=0
    # orbit_counts = df.loc[df['Orbit'].value_counts()!=4,df['Orbit']]
    # print(orbit_counts)
    NumCrossings = []
    counter = 0
    for i in range(1,len(df.index)):
        if df.Orbit[i] != df.Orbit[i-1]:
            NumCrossings.append(counter)
            counter = 0
            counter += 1
        else:
            counter += 1
    NumCrossings = np.asarray(NumCrossings)
    # Print orbit number where there is not 4 crossings
    weird_crossings = np.where(NumCrossings != 4)
    print(len(weird_crossings[0]))
    return NumCrossings, weird_crossings

def time_in_sheath(df,df_sun):
    '''Histogram of time in sheath durations: run df through orbits first''' 
    #mp_out(end)-bs_out(start)
    #bs_in(end)-mp_in(start)

    def sheath(df):
        Dur_out=[]
        Dur_in = []
        for i in df.index[:-1]:
            if df.Type[i] == 'mp_out' and df.Type[i+1] == 'bs_out':
                Dur_out.append((df.start[i+1]-df.end[i]).total_seconds())
            elif df.Type[i] == 'bs_in' and df.Type[i+1] == 'mp_in':
                Dur_in.append((df.start[i+1]-df.end[i]).total_seconds())
        return Dur_in,Dur_out
    Dur_in_p, Dur_out_p = sheath(df)
    Dur_in_s, Dur_out_s = sheath(df_sun)
    bins = 100
    plt.hist(Dur_in_p, bins=bins, alpha=0.5,label='Sheath_in_p',color='k',histtype='step')
    plt.hist(Dur_out_p, bins=bins,alpha=0.5,label='Sheath_out_p',color='b',histtype='step')
    plt.hist(Dur_in_s, bins=bins, alpha=0.5,label='Sheath_in_s',color='orange',histtype='step')
    plt.hist(Dur_out_s, bins=bins,alpha=0.5,label='Sheath_out_s',color='darkred',histtype='step')
    plt.legend()
    #plt.yscale('log')
    plt.xlabel('Duration of time in sheath (s)')
    return 


def save_largest(df,sun = False, philpott = False):
    df = df.nlargest(10, 'Interval')
    minute = datetime.timedelta(minutes=10)
    start_time = []
    end_time = []
    for i in df.start:
        i = i-minute
        parsed_datetime = datetime.datetime.strptime(str(i), "%Y-%m-%d %H:%M:%S")
        output_datetime_str = parsed_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        start_time.append(str(output_datetime_str))
    for i in df.end:
        i = i+minute
        parsed_datetime = datetime.datetime.strptime(str(i), "%Y-%m-%d %H:%M:%S")
        output_datetime_str = parsed_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        end_time.append(str(output_datetime_str))
    for i in range(0,10):
        if sun == True:
            mag.mag_time_series(start_date=start_time[i],end_date=end_time[i],sun=True, save = True, num = i)
        if philpott == True:
            mag.mag_time_series(start_date=start_time[i],end_date=end_time[i],philpott=True, save = True, num = i)
        


# combining all 60 second resolution data into one .csv file with 6 min resolution to find orbits  
import os
def find_tab_files(root_dir):
    """Recursively find all .TAB files in root_dir and its subdirectories."""
    tab_files = []
    for path, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('60_V08.TAB'):
                tab_files.append(os.path.join(path, file))
    return tab_files

def combine_tab_files(tab_files, output_file):
    """Combine multiple .TAB files into one CSV file using np.genfromtxt."""
    combined_data = []
    
    for file in tab_files:
        data = np.genfromtxt(file, skip_header=0)
        x = np.array(data)[::10]
        combined_data.append(x)
    
    # Concatenate all the data arrays
    combined_data = np.concatenate(combined_data)
    
    # Save to a new CSV file
    np.savetxt(output_file, combined_data, delimiter=',')
    
def read_in_all():
    from datetime import datetime, timedelta

    # Read the CSV file into a DataFrame
    df = pd.read_csv('all.csv', header=None)
    df.columns = ['year', 'day_of_year', 'hour', 'second', 'col5', 'col6', 'col7', 'ephx', 'ephy', 'ephz','col11','col12','col13','col14','col15','col16']
    # Combine the time components into a single datetime column
    def combine_time_components(row):
        year = int(row['year'])
        day_of_year = int(row['day_of_year'])
        hour = int(row['hour'])
        second = float(row['second'])
        
        # Combine the time components into a datetime object
        date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour, seconds=second)
        return date

    # Apply the function to each row to create the 'time' column
    df['time'] = df.apply(combine_time_components, axis=1)

    # Select the relevant columns
    df = df[['time', 'ephx', 'ephy', 'ephz']]
    date_string = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    ephx = df['ephx']
    ephy = df['ephy']
    msm_ephx = []
    msm_ephy = []
    for i in tqdm.tqdm(range(0,len(date_string))):
        phi=load_messenger_mag.get_aberration_angle(date_string[i])
        new_ephx = ephx[i]*np.cos(phi)-ephy[i]*np.sin(phi)
        new_ephy = ephx[i]*np.sin(phi)+ephy[i]*np.cos(phi)
        msm_ephx.append(new_ephx)
        msm_ephy.append(new_ephy)
    
    df['ephem_x'] = msm_ephx
    df['ephem_y'] = msm_ephy
    # Display the resulting DataFrame
    return df

def orbit(df_all,df_crossing):
    #Uncomment below for first run
    r_all = np.sqrt(df_all.ephx.to_numpy()**2 + df_all.ephy.to_numpy()**2 + \
                        df_all.ephz.to_numpy()**2)
    from scipy.signal import find_peaks
    peaks=find_peaks(r_all)
    orbits = 0
    df_crossing['Orbit'] = len(peaks[0]-1)
    for p in tqdm.tqdm(range(len(peaks[0])-1)):
        orbit_range = [peaks[0][p],peaks[0][p+1]]
        time_range = [df_all.time.iloc[orbit_range[0]],df_all.time.iloc[orbit_range[1]]]
        df_crossing.loc[(df_crossing['start'] <= time_range[1]) & (df_crossing['start'] >= time_range[0]), 'Orbit'] = int(orbits)
        orbits+=1
    print(df_crossing)
    return df_crossing


def analyse_largest_mp_crossings_sheath(df):
    #input is crossing list
    df = df[(df.Type == 'mp_in') | (df.Type == 'mp_out')]
    minutes = datetime.timedelta(minutes=10)
    df['Interval'] = (df.end-df.start).dt.total_seconds()
    df = df.nlargest(10, 'Interval')
    for i in df.index:
        if df.loc[i,'Type'] == 'mp_in':
            parsed_datetime_s = datetime.datetime.strptime(str(df.loc[i,'start']-minutes), "%Y-%m-%d %H:%M:%S")
            start_str = parsed_datetime_s.strftime("%Y-%m-%d-%H-%M-%S")
            parsed_datetime_e = datetime.datetime.strptime(str(df.loc[i,'start']), "%Y-%m-%d %H:%M:%S")
            end_str = parsed_datetime_e.strftime("%Y-%m-%d-%H-%M-%S")
            df_info = mag.mag_time_series(start_date=start_str,end_date=end_str,plot=False)
            #Add to new column
            df.loc[i,'avg_Bx_10'] = (df_info['mag_x'].mean())
            df.loc[i,'std_Bx_10'] = (df_info['mag_x'].std())
            half = int(len(df_info)/2)
            df_info_5 = df_info.iloc[half:]
            df.loc[i,'avg_Bx_5'] = (df_info_5['mag_x'].mean())
            df.loc[i,'std_Bx_5'] = (df_info_5['mag_x'].std())
        if df.loc[i,'Type'] == 'mp_out':
            parsed_datetime_s = datetime.datetime.strptime(str(df.loc[i,'end']), "%Y-%m-%d %H:%M:%S")
            start_str = parsed_datetime_s.strftime("%Y-%m-%d-%H-%M-%S")
            parsed_datetime_e = datetime.datetime.strptime(str(df.loc[i,'end']+minutes), "%Y-%m-%d %H:%M:%S")
            end_str = parsed_datetime_e.strftime("%Y-%m-%d-%H-%M-%S")
            df_info = mag.mag_time_series(start_date=start_str,end_date=end_str,plot=False)
            #Add to new column
            df.loc[i,'avg_Bx_10'] = (df_info['mag_x'].mean())
            df.loc[i,'std_Bx_10'] = (df_info['mag_x'].std())
            half = int(len(df_info)/2)
            df_info_5 = df_info.iloc[:-half]
            df.loc[i,'avg_Bx_5'] = (df_info_5['mag_x'].mean())
            df.loc[i,'std_Bx_5'] = (df_info_5['mag_x'].std())


    return df

        
        #print(df_info)

# def compare_boundaries(df1, df2, dt=30):

#     '''
#     Will be code to cross compare the two lists (df_sun & df_p)
#     How to implement?
#     Find events with approx same start time
#     Find difference in x_msm and rho_msm
#     Find events identified in one list but not the other
#     '''
#     #First plot all events identified in one list but not the other
#     df1 = df1[df1["start"].index < 16000]
#     df2 = df2[df2["start"].index < 16000]

#     #Function that creates new df with elements unique to df2
#     #Checks if start time is same +/- dt
#     def only_one_df(df1,df2,dt=dt):
#         temp=[]
#         j=0
#         for i in df1["start"]:
#             check_time=[i-datetime.timedelta(seconds=dt),i+datetime.timedelta(seconds=dt)]
#             x = df2.index[(df2["start"] >= check_time[0])
#                         & (df2["start"] <= check_time[1])]
#             if len(x) == 0:
#                 temp.append(j)
#             j+=1
#             print(f"{j}/{len(df1["start"])}", end = '\r')
#         print(temp)
#         df2_only = df2[df2.index.isin(temp)]

#         return df2_only

#     #Alt method: check if crossing occurs within one from other list
#     def only_one_df_2(df1,df2):
#         temp=[]
#         j=0
#         for i in range(len(df1)):
#             tempdf=df1.loc[i]
#             x = df1.index[((tempdf['start'] >= df2['start'])
#                         & (tempdf['start'] <= df2['end'])) | ((tempdf['end'] >= df2['start'])
#                         & (tempdf['end'] <= df2['end'])) | ((df2['start'] >= tempdf['start'])
#                         & (df2['start'] <= tempdf['end'])) | ((df2['end'] >= tempdf['start'])
#                         & (df2['end'] <= tempdf['end']))]
#             if len(x) == 0:
#                 temp.append(j)
#             j+=1
#             print(f"{j}/{len(df1["start"])}", end = '\r')
#         print(temp)
#         df1_only = df1[df1.index.isin(temp)]

#         return df1_only

#     df1_only = only_one_df(df2,df1)
#     df2_only = only_one_df(df1,df2)

#     print(len(df1_only))
#     print(len(df2_only))

#     #plot_boundary_locations(df1_only)
#     #plot_boundary_locations(df2_only)

#     return df1_only,df2_only

# #Function to give sample of where list differ in start time by more than t seconds
# def compare_lists(df1,df2,t):
#     years = datetime.datetime(2014,12,30)
#     df1_only, df2_only = compare_boundaries(df1, df2,t)
#     df1_only = df1_only[df1_only["start"] > years]
#     df2_only = df2_only[df2_only["start"] > years]

#     print(df2_only.sample(n=5))

# #Function to plot histogram of time difference between crossings
# def compare_list_hist(df1,df2,t):
#     df1_only, df2_only = compare_boundaries(df1, df2,t)
#     df1_merge = df1.merge(df1_only,how='left',indicator=True)
#     df1_both = df1_merge[df1_merge['_merge'] == 'left_only'].drop(columns=['_merge'])
#     df2_merge = df2.merge(df2_only,how='left',indicator=True)
#     df2_both = df2_merge[df2_merge['_merge'] == 'left_only'].drop(columns=['_merge'])

#     df1_both=df1_both.reset_index()
#     df2_both=df2_both.reset_index()
#     print(df1_both['start']-df2_both['start'])

#     return

# def orbits_old(df):
#     '''
#     Assign orbit number to each crossing. Calling 1 orbit bs_in to bs_out, assuming not every orbit has bs but has mp_in.
#     '''
#     df['Orbit'] = 0
#     orbit_number = -1
#     for i in df.index:
#         if df.Type[i] == "mp_in":
#             orbit_number += 1
#             if df.Type[i-1] == 'bs_in':
#                 df.loc[i-1,'Orbit'] = orbit_number
#         if orbit_number == -1:
#             df.loc[i,'Orbit'] = orbit_number+1
#         else:
#             df.loc[i,'Orbit'] = orbit_number
#     return orbit_number

# def orbits_from_list(df):
#     '''Function to assign orbit number to each crossing based on peri-to-peri'''
#     '''Use eph data? Thats a lot of data! Could use 60 sec resolution so ~100mB'''
#     r_all = np.sqrt(df.start_x_msm.to_numpy()**2 + df.start_y_msm.to_numpy()**2 + \
#                     df.start_z_msm.to_numpy()**2)
#     t_all = df.start
#     from scipy.signal import find_peaks
    
#     peaks, _ = find_peaks(-r_all)

#     # Filter peaks based on time separation
#     min_separation = datetime.timedelta(hours=6)  # Minimum separation in time
#     filtered_peaks = []
#     print(len(peaks))
#     for peak in peaks:
#         if len(filtered_peaks) == 0 or t_all[peak] - t_all[filtered_peaks[-1]] >= min_separation:
#             filtered_peaks.append(peak)
#     peaks=filtered_peaks
#     print(len(peaks))
#     orbit = 1
#     df['Orbit'] = 0
#     for p in range(len(peaks)-1):
#         orbit_range = [peaks[p],peaks[p+1]]
#         df.loc[peaks[p]:peaks[p+1],'Orbit'] = orbit
#         orbit+=1
#         time_range = [df.start.iloc[orbit_range[0]],df.start.iloc[orbit_range[1]]]
#         if p == len(peaks)-2:
#             df.loc[peaks[-1]+1:,'Orbit'] = orbit
#     return df
