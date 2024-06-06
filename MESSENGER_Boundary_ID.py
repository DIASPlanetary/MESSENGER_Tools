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
    
philpott_file = '/Users/bowersch/Downloads/jgra55678-sup-0002-table_si-s01.csv'



''' Link to download Sun boundary list: 
    
     https://zenodo.org/records/8298647
    
    Need to download separate .txt files for each boundary crossing type.
    
    Save these into a folder on your machine.

    Specify location of folder: '''
    
Sun_crossings_folder = '/Users/bowersch/Desktop/MESSENGER Data/Weijie Crossings/'


'''

The Sun crossing list only has the times of the boundaries, not their location.

I have included a .csv file in the github that has both the boundary times and locations 
of the Sun list (Sun_Boundaries_with_Eph.csv).

Specify the location of this .csv file here:

'''

Sun_csv = '/Users/bowersch/Desktop/Python_Code/MESSENGER_Boundary_Work/Sun_Boundaries_with_Eph.csv'


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

import numpy as np

import pandas as pd

import datetime

import ephem

import matplotlib.pyplot as plt

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
        date += datetime.timedelta(days=float(day_of_year) - 1, hours=float(hour), minutes=float(minute),seconds=float(second))
        
        return date
    
    dt = np.array([create_datestring(df_boundaries.Yr_pass.iloc[p],\
                                                  df_boundaries.Day_orbit.iloc[p],\
                                                  df_boundaries.Hour.iloc[p],\
                                                  df_boundaries.Minute.iloc[p],\
                                                  round(df_boundaries.Second.iloc[p]))\
                                for p in range(len(df_boundaries))])
        
    df_boundaries['time'] = dt
    
    Z_MSM = df_boundaries['Z_MSO (km)']/2440-.19
    
    X_MSM = np.array([])
    
    Y_MSM = np.array([])
    
    cross_string = np.array([])
    
    cross_strings = np.array(['err','bs_in_1','bs_in_2','mp_in_1','mp_in_2',
                               'mp_out_1','mp_out_2','bs_out_1','bs_out_2','gap_1','gap_2'])
    
    def rotate_into_msm(x,y,z,time):
        from trying3 import get_aberration_angle
        
        #Aberration:
            
        def get_aberration_angle(date):
            
            import numpy as np
            
            def get_mercury_distance_to_sun(date):
                # create a PyEphem observer for the Sun
                
                
                j = ephem.Mercury()
                
                j.compute(date,epoch='1970')

                distance_au=j.sun_distance
                
   
                return distance_au
            
            
            # Estimate instantaneous orbital velocity of Mercury:
                
            r=get_mercury_distance_to_sun(date)*1.496E11
            
            a=57909050*1000.
            
            M=1.9891E30
            
            G=6.67430E-11
            
            v=np.sqrt(G*M*(2./r-1./a))
            
            # Calculate aberration angle assuming 400 km/s sw speed
            
            alpha=np.arctan(v/400000)
            
            return alpha
            
        phi=get_aberration_angle(time)
        
        x_msm=x*np.cos(phi)-y*np.sin(phi)
        
        y_msm=y*np.sin(phi)+y*np.cos(phi)
        
        return x_msm,y_msm,z
    
    
    
    
    for i in range(len(df_boundaries)):
        
    
        X_MSM_1, Y_MSM_1, Z_MSM_2 = rotate_into_msm(df_boundaries['X_MSO (km)'].iloc[i]/2440,
                                         df_boundaries['Y_MSO (km)'].iloc[i]/2440,
                                         Z_MSM[i],
                                         df_boundaries.time.iloc[i])
        
        X_MSM = np.append(X_MSM,X_MSM_1)
        
        Y_MSM = np.append(Y_MSM,Y_MSM_1)
        
        cross_string = np.append(cross_string,cross_strings[df_boundaries['Boundary number'].iloc[i]])
            
            
            
        
            
        
    df_boundaries[['X_MSM','Y_MSM','Z_MSM']] = np.stack((X_MSM,Y_MSM,Z_MSM),axis=1)
    
    df_boundaries['Cross_Type'] = cross_string
    
    stacked_df_data =pd.DataFrame( {'start':[np.nan],
                       'end':[np.nan],
                       'start_x_msm':[np.nan],
                       'end_x_msm':[np.nan],
                       'start_y_msm':[np.nan],
                       'end_y_msm':[np.nan],
                       'start_z_msm':[np.nan],
                       'end_z_msm':[np.nan],
                       'Type':[np.nan]})
    
    cross_strings = ['bs_in','bs_out','mp_in','mp_out','gap']
    
    for i in cross_strings:

        s = df_boundaries[(df_boundaries.Cross_Type == i+'_1')]
        e = df_boundaries[(df_boundaries.Cross_Type == i+'_2')]
        
        data = {'start': s.time.to_numpy(),
                'end': e.time.to_numpy(),
                'start_x_msm': s.X_MSM.to_numpy(),
                'end_x_msm':e.X_MSM.to_numpy(),
                'start_y_msm':s.Y_MSM.to_numpy(),
                'end_y_msm':e.Y_MSM.to_numpy(),
                'start_z_msm':s.Z_MSM.to_numpy(),
                'end_z_msm':e.Z_MSM.to_numpy(),
                'Type':i}
        
        stacked_df_data = pd.concat([stacked_df_data,pd.DataFrame(data)])
        
    
        
    #stacked_df_data = stacked_df_data.drop(0).reset_index(drop=True)
    
    stacked_df_data = stacked_df_data.sort_values('start',ignore_index=True)
    
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
        
        x_in = np.loadtxt(file,usecols=(0,1,2,3,4,5))
        x_out = np.loadtxt(file,usecols=(6,7,8,9,10,11))
        date_in = np.array([])
        date_out = np.array([])
        
        # Correct for annoying run overs (minutes = 60, etc.)
        for i in range(np.size(x_in[:,0])):
            
            if int(np.floor(x_in[i,5])) >= 60:
                x_in[i,5]=0.0
                x_in[i,4]=x_in[i,4]+1
                
            if int(np.floor(x_out[i,5])) >= 60:
                x_out[i,5]=0.0
                x_out[i,4]=x_out[i,4]+1
                
            if int(np.floor(x_out[i,5])) < 0:
                x_out[i,5]=59
                x_out[i,4]=x_out[i,4]-1
                
                if x_out[i,4]<0:
                    x_out[i,3]=x_out[i,3]-1
                    x_out[i,4]=59
                
            
            if int(np.floor(x_in[i,5])) < 0:
                x_in[i,5]=59
                x_in[i,4]=x_in[i,4]-1
                if x_in[i,4]<0:
                    x_in[i,3]=x_in[i,3]-1
                    x_in[i,4]=59
            
            
            def convert_to_datetime(date_string):
                import datetime
                date_obj=datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
                
                return date_obj
            
            date_string_in = str(int(np.floor(x_in[i,0])))+'-'+str(int(np.floor(x_in[i,1])))+\
                '-'+str(int(np.floor(x_in[i,2])))+' '+str(int(np.floor(x_in[i,3])))+\
                              ':'+str(int(np.floor(x_in[i,4])))+':'+str(int(np.floor(x_in[i,5])))
                              
            date_datetime_in = convert_to_datetime(date_string_in)
            
            date_in=np.append(date_in,date_datetime_in)
                              
                              
            date_string_out = str(int(np.floor(x_out[i,0])))+'-'+str(int(np.floor(x_out[i,1])))+\
                '-'+str(int(np.floor(x_out[i,2])))+' '+str(int(np.floor(x_out[i,3])))+\
                               ':'+str(int(np.floor(x_out[i,4])))+':'+str(int(np.floor(x_out[i,5])))
                               
            date_datetime_out = convert_to_datetime(date_string_out)


            date_out=np.append(date_out,date_datetime_out)
                                                                
            
            
        date=np.array([date_in,date_out])
        
        return date
    
    file_mp_in=scf+'MagPause_In_Time_Duration__public_version_WeijieSun_20230829.txt'
    file_mp_out=scf+'MagPause_Out_Time_Duration_public_version_WeijieSun_20230829.txt'
    file_bs_in=scf+'Bow_Shock_In_Time_Duration__public_version_WeijieSun_20230829.txt'
    file_bs_out=scf+'Bow_Shock_Out_Time_Duration_public_version_WeijieSun_20230829.txt'
    
    mp_in=convert_Sun_txt_to_date(file_mp_in)
    mp_out=convert_Sun_txt_to_date(file_mp_out)
    bs_in=convert_Sun_txt_to_date(file_bs_in)
    bs_out=convert_Sun_txt_to_date(file_bs_out)
    
    def generate_crossing_dataframe(cross,typ,eph=False):
        import numpy as np
        import pandas as pd

        
        cross_start=cross[0,:]
        
        cross_end=cross[1,:]
        
        cross_df=pd.DataFrame(data={'start':cross_start,'end':cross_end})
        
        cross_df['Type']=typ
        
        
        return cross_df
    
    mi=generate_crossing_dataframe(mp_in,'mp_in')
    
    mo=generate_crossing_dataframe(mp_out,'mp_out')
    
    bi=generate_crossing_dataframe(bs_in,'bs_in')
    
    bo=generate_crossing_dataframe(bs_out,'bs_out')
    
    crossings=[mi,mo,bi,bo]
    
    cc=pd.concat(crossings)
    
    c=cc.sort_values('start')
    
    return c


def read_in_Sun_csv(Sun_csv):
    
    df_Sun = pd.read_csv(Sun_csv)
    
    return df_Sun
    

def plot_boundary_locations(df):
    
    '''Create a plot of Mercury and the location of the boundaries in cylindrical coordinates
    
    Input a dataframe loaded by read_in_Philpott_list or read_in_Sun_list
    
    Outputs a plot of the magnetopause and bow shock boundaries onto a cylindrical map of
    Mercury with dashed lines for the nominal magnetopause and bow shock shapes determined
    by Winslow et al., 2013
    
    '''
    
    #Plot Mercury
    
    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)-0.2
    
    fig, ax1 = plt.subplots(1)
    
    # Plot the circle in all 3 plots
    ax1.plot(x, y, color='gray')

    
    ax1.set_xlabel("$X_{MSM\'}$ ($R_M$)",fontsize=20)
    
    ax1.set_ylabel("\u03C1$_{MSM\'}$ ($R_M$)",fontsize=20)
    
    ax1.tick_params(labelsize=20)
    
    
    def plot_mp_and_bs(ax1):
        
        ''' Plot Nominal Magnetopause and Bow Shock Location from Winslow 2013'''
        
        y_mp=np.linspace(-100,100,100)
        z_mp=np.linspace(-100,100,100)
        x_mp=np.linspace(-10,10,100)
        
        rho=np.sqrt(y_mp**2+(z_mp)**2)
 
        phi=np.arctan2(rho,x_mp)

        Rss=1.45
        
        alpha=0.5
        
        phi2 = (np.linspace(0,2*np.pi,100))
        
        rho=Rss*(2/(1+np.cos(phi2)))**(alpha)
        
        xmp=rho*np.cos(phi2)
        
        ymp=rho*np.sin(phi2)

        
        ax1.plot(xmp,ymp,color='black',linestyle='--',linewidth=3)
        
        psi=1.04

        p=2.75

        L=psi*p

        x0=.5

        phi = (np.linspace(0,2*np.pi,100))
        rho = L/(1. + psi*np.cos(phi))

        xshock = x0 + rho*np.cos(phi)
        yshock = rho*np.sin(phi)
        
        ax1.plot(xshock,yshock,color='black',linestyle='--',linewidth=3)
        
    plot_mp_and_bs(ax1)

    # Color the left hemisphere red and the right hemisphere gray
    ax1.fill_between(x, y, where=x<0, color='black', interpolate=True)
    #Set equal aspect so Mercury is circular
    ax1.set_aspect('equal', adjustable='box')
    
    # Set the limits of the plot in all 3 plots
    ax1.set_xlim([-5, 3])
    ax1.set_ylim([0, 4])
    
    df_mp = df[((df.Type == 'mp_in') | (df.Type =='mp_out'))]
    
    df_bs = df[((df.Type == 'bs_in') | (df.Type =='bs_out'))]

    def plot_mean_locations(df,cr,lb):
        
    
        mean_x = np.mean(df[['start_x_msm','end_x_msm']],axis=1)
        
        mean_y = np.mean(df[['start_y_msm','end_y_msm']],axis=1)
        
        mean_z = np.mean(df[['start_z_msm','end_z_msm']],axis=1) 
        

        
        r_msm = np.sqrt(mean_y**2+mean_z**2)
        
        ax1.scatter(mean_x, r_msm,s=.1,color=cr,label = lb)
        
    plot_mean_locations(df_mp,'indianred','MP')
    plot_mean_locations(df_bs,'mediumturquoise','BS')
    
    ax1.legend()
        
    
    

    
    
