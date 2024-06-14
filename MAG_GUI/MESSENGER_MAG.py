import pyspedas
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import csv
import matplotlib.dates as mdates
import MESSENGER_MAG
import numpy as np
import urllib.request
import pandas as pd

# Define global variables: MESSENGER data path. The default is the folder where the current script resides.
current_path = os.path.abspath(__file__)
# Get the path to the folder where the current script file resides
MESSENGER_data_folder = os.path.dirname(current_path)  # To customize the path, modify it here
# The data folder structure is：MESSENGER_data_folder +'/MESSENGER_Data/MAG/Science_MAG/NPY/'

def Read_MAG_TABdata(filename, mode='01'):
    #Read TAB files locally and convert them to npy files (improved reading speed), default data mode is' 01 '(1s resolution data)
    import numpy as np
    import PyFileIO as pf
    if mode == '0.05':
        pdsdtype = [('Year', 'int32'),
                    ('DOY', 'int32'),
                    ('Hour', 'int32'),
                    ('Min', 'int32'),
                    ('Sec', 'object'),
                    ('MET', 'float32'),
                    ('Xmso', 'float32'),
                    ('Ymso', 'float32'),
                    ('Zmso', 'float32'),
                    ('Bx', 'float32'),
                    ('By', 'float32'),
                    ('Bz', 'float32')]
    else:
        pdsdtype = [('Year', 'int32'),
                    ('DOY', 'int32'),
                    ('Hour', 'int32'),
                    ('Min', 'int32'),
                    ('Sec', 'object'),
                    ('MET', 'float32'),
                    ('Sample_interval', 'int32'),
                    ('Xmso', 'float32'),
                    ('Ymso', 'float32'),
                    ('Zmso', 'float32'),
                    ('Bx', 'float32'),
                    ('By', 'float32'),
                    ('Bz', 'float32'),
                    ('dBx', 'float32'),
                    ('dBy', 'float32'),
                    ('dBz', 'float32')]
    data = pf.ReadASCIIData(filename, False, dtype=pdsdtype)
    n = data.size
    outdtype = [('Year', 'int32'),
                ('DOY', 'int32'),
                ('Hour', 'int32'),
                ('Min', 'int32'),
                ('Sec', 'object'),
                ('MET', 'float32'),
                ('Time', 'datetime64[ms]'),
                ('Xmso', 'float32'),
                ('Ymso', 'float32'),
                ('Zmso', 'float32'),
                ('Bx', 'float32'),
                ('By', 'float32'),
                ('Bz', 'float32')]
    out = np.recarray(n, outdtype)
    out.Year = data.Year
    out.DOY = data.DOY
    out.Hour = data.Hour
    out.Min = data.Min
    out.Sec = data.Sec
    out.MET = data.MET
    for i in range(0, n):
        time_str = doy_to_date(data.Year[i], data.DOY[i]) + ' ' + str(data.Hour[i]).zfill(2) + ':' + str(
            data.Min[i]).zfill(2) + ':' + data.Sec[i]
        out.Time[i] = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    out.Xmso = data.Xmso / 2440.0
    out.Ymso = data.Ymso / 2440.0
    out.Zmso = data.Zmso / 2440.0
    out.Bx = data.Bx
    out.By = data.By
    out.Bz = data.Bz
    # By default, the NYP data folder is automatically created under the current script folder, and the data is saved in the folder. Of course, you can also modify the path yourself.
    # Gets the path to the current script file

    if mode == '0.05':
        npyfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Science_MAG/NPY/'
        if os.path.isdir(npyfilepath):
            np.save(npyfilepath + 'MAGMSOSCI' + str(out.Year[0])[2:4] + str(out.DOY[0]).zfill(3)  + '_V08.npy', out)
        else:
            os.makedirs(npyfilepath)
            np.save(npyfilepath + 'MAGMSOSCI' + str(out.Year[0])[2:4] + str(out.DOY[0]).zfill(3)  + '_V08.npy', out)
    else:
        npyfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Reduced_MAG/NPY/'
        if os.path.isdir(npyfilepath):
            np.save(npyfilepath + 'MAGMSOSCIAVG' + str(out.Year[0])[2:4] + str(out.DOY[0]).zfill(3)  + '_' + mode + '_V08.npy',
                    out)
        else:
            os.makedirs(npyfilepath)
            np.save(npyfilepath + 'MAGMSOSCIAVG' + str(out.Year[0])[2:4] + str(out.DOY[0]).zfill(3)  + '_' + mode + '_V08.npy',
                    out)

def day_of_year(year, month, day):
    # The conversion date is the day of the year
    from datetime import datetime
    date = datetime(year, month, day)
    start_of_year = datetime(year, 1, 1)
    return (date - start_of_year).days + 1

def doy_to_date(year, day_of_year):
    # How many days of the year are converted to dates
    # The first day of the specified year
    start_of_year = datetime(year, 1, 1)
    # Calculate the target date by increasing the number of days
    target_date = start_of_year + timedelta(days=int(day_of_year) - 1)
    # Format the date as YYYY-MM-DD
    return target_date.strftime('%Y-%m-%d')

def Read_Mag_npydata(Time_Range, mode='01'):
    #读取本地npy文件，若本地存在对应时间npy文件，则进行读取，若不存在则PDS自动下载到本地
    from MESSENGER_MAG import day_of_year
    DOY_start = day_of_year(int(Time_Range[0][0:4]), int(Time_Range[0][5:7]), int(Time_Range[0][8:10]))
    DOY_end = day_of_year(int(Time_Range[1][0:4]), int(Time_Range[1][5:7]), int(Time_Range[1][8:10]))
    #检查是否存在npy数据文件夹,若不存在则创立npy数据文件夹
    if mode == '0.05':
        npyfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Science_MAG/NPY/'  #0.05s分辨率数据文件夹
        if not os.path.isdir(npyfilepath):
            os.makedirs(npyfilepath)
    else:
        npyfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Reduced_MAG/NPY/'  #01、05、10、60s分辨率数据文件夹
        if not os.path.isdir(npyfilepath):
            os.makedirs(npyfilepath)
    #若时间范围跨多个文件则将连接多个文件
    MAG = {}  #创建连接多个npy文件数据的空字典
    for i in range(DOY_start, DOY_end + 1):
        if mode == '0.05':
            npyfilename = 'MAGMSOSCI' + Time_Range[0][2:4] + str(i).zfill(3) + '_V08.npy'
        else:
            npyfilename = 'MAGMSOSCIAVG' + Time_Range[0][2:4] + str(i).zfill(3) + '_' + mode + '_V08.npy'
        npyfull_path = os.path.join(npyfilepath, npyfilename)
        # 检查文件是否存在
        if os.path.exists(npyfull_path):
            MAG_npy = np.load(npyfull_path, allow_pickle=True)
            print("npy.File loaded successfully.")
        else:
            print("npy.File does not exist,and will attempt to convert TAB. data to npy. data.")
            if mode == '0.05':
                tabfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Science_MAG/TAB/'  # 0.05s分辨率数据文件夹
                if not os.path.isdir(npyfilepath):
                    os.makedirs(npyfilepath)
                tabfilename = 'MAGMSOSCI' + Time_Range[0][2:4] + str(i).zfill(3) + '_V08.TAB'
            else:
                tabfilepath = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Reduced_MAG/TAB/'  # 01、05、10、60s分辨率数据文件夹
                if not os.path.isdir(tabfilepath):
                    os.makedirs(npyfilepath)
                tabfilename = 'MAGMSOSCIAVG' + Time_Range[0][2:4] + str(i).zfill(3) + '_' + mode + '_V08.TAB'
            tabfull_path = os.path.join(tabfilepath, tabfilename)
            if os.path.exists(tabfull_path):
                print("tab.File loaded successfully and will be transfered to npy.file.")
                MESSENGER_MAG.Read_MAG_TABdata(tabfull_path, mode)
                MAG_npy = np.load(npyfull_path, allow_pickle=True)
            else:
                print("tab.File does not exist,and will be download from PDS.")
                MESSENGER_MAG.Download_MESSENGER_Mag_Data(int(Time_Range[0][0:4]), i, mode)
                MESSENGER_MAG.Read_MAG_TABdata(tabfull_path, mode)
                MAG_npy = np.load(npyfull_path, allow_pickle=True)
        for name in MAG_npy.dtype.names:
            # 如果数组名已经存在于 merged_arrays 中，则将当前数组与已存在的数组合并
            if name in MAG:
                MAG[name] = np.concatenate((MAG[name], MAG_npy[name]))
            # 如果数组名不存在于 merged_arrays 中，则将当前数组添加到 merged_arrays 中
            else:
                MAG[name] = MAG_npy[name]
    Time_range_s = datetime.strptime(Time_Range[0], "%Y-%m-%d %H:%M:%S")#此处可更改
    Time_range_e = datetime.strptime(Time_Range[1], "%Y-%m-%d %H:%M:%S")
    index = np.where((MAG['Time'] > Time_range_s) & (MAG['Time'] < Time_range_e))
    returndtype = [('Year', 'int32'),
                   ('DOY', 'int32'),
                   ('Hour', 'int32'),
                   ('Min', 'int32'),
                   ('Sec', 'object'),
                   ('MET', 'float32'),
                   ('Time', 'datetime64[ms]'),
                   ('Xmso', 'float32'),
                   ('Ymso', 'float32'),
                   ('Zmso', 'float32'),
                   ('Bx', 'float32'),
                   ('By', 'float32'),
                   ('Bz', 'float32'),
                   ('t_tag', 'U')]
    returndata = np.recarray(len(index[0]), returndtype)
    returndata.Year = MAG['Year'][index[0]]
    returndata.DOY = MAG['DOY'][index[0]]
    returndata.Hour = MAG['Hour'][index[0]]
    returndata.Min = MAG['Min'][index[0]]
    returndata.Sec = MAG['Sec'][index[0]]
    returndata.MET = MAG['MET'][index[0]]
    returndata.Time = MAG['Time'][index[0]]
    returndata.Xmso = MAG['Xmso'][index[0]]
    returndata.Ymso = MAG['Ymso'][index[0]]
    returndata.Zmso = MAG['Zmso'][index[0]]
    returndata.Bx = MAG['Bx'][index[0]]
    returndata.By = MAG['By'][index[0]]
    returndata.Bz = MAG['Bz'][index[0]]
    return returndata

# Define global variables to hold time ranges and indexes
time_range_index = 0
def MAG_GUI(time_ranges=None, mode='01', time_list=None):
    # Magnetic field data interface, you can set the time range list (time_list = None at this time)
    # If time_list='MP_Sun', assign Sun's magnetopause event list to time_ranges
    # If time_list='BS_Sun', assign Sun's bow shock event list to time_ranges

    if time_list=='MP_Sun':
        mplist = np.load(MESSENGER_data_folder+'/mplist.npy',allow_pickle=True)
        # Initializes the time ranges list
        time_ranges = []
        # Traverse the mplist, combining start and end into tuples to add to time_ranges
        for record in mplist:
            start = record['start']
            end = record['end']
            # The start time and end time of parsing are datetime objects
            start_time = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')

            # Expand the time range: Add 2 minute each time
            extended_start = start_time - timedelta(minutes=2)
            extended_end = end_time + timedelta(minutes=2)

            # Adds the extended time range to the time ranges
            time_ranges.append((extended_start.strftime('%Y-%m-%d %H:%M:%S'), extended_end.strftime('%Y-%m-%d %H:%M:%S')))

    if time_list=='BS_Sun':
        bslist = np.load(MESSENGER_data_folder+'/bslist.npy',allow_pickle=True)
        # Initializes the time ranges list
        time_ranges = []
        # Iterate over the bslist, combining start and end into tuples to add to time_ranges
        for record in bslist:
            start = record['start']
            end = record['end']
            # 解析开始时间和结束时间为 datetime 对象
            start_time = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')

            # 扩展时间范围：前后各增加1分钟
            extended_start = start_time - timedelta(minutes=2)
            extended_end = end_time + timedelta(minutes=2)

            # Adds the extended time range to the time ranges
            time_ranges.append((extended_start.strftime('%Y-%m-%d %H:%M:%S'), extended_end.strftime('%Y-%m-%d %H:%M:%S')))

    global time_range_index
    current_time_range = time_ranges[time_range_index]
    # Create main window
    root = tk.Tk()
    root.title("Interactive Plot")
    root.geometry("1100x800")
    root.configure(bg='#f0f0f0')

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', font=('Helvetica', 12), padding=10, background='#4CAF50', foreground='white')
    style.map('TButton', background=[('active', '#45a049')])
    style.configure('TLabel', font=('Helvetica', 12), background='#f0f0f0')
    style.configure('TFrame', background='#f0f0f0')

    mag_mso = MESSENGER_MAG.Read_Mag_npydata(current_time_range, mode)
    bt = np.sqrt(mag_mso.Bx ** 2 + mag_mso.By ** 2 + mag_mso.Bz ** 2)

    # Create plot area
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.subplots_adjust(hspace=0)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Initial plot data
    ax1.plot(mag_mso.Time, bt, 'black', linewidth=0.5)
    ax1.set_title("MESSENGER MAG", fontsize=16)
    ax1.set_ylabel('Bt', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])

    ax2.plot(mag_mso.Time, mag_mso.Bx, linewidth=0.5, label='Bx')
    ax2.plot(mag_mso.Time, mag_mso.By, linewidth=0.5, label='By')
    ax2.plot(mag_mso.Time, mag_mso.Bz, linewidth=0.5, label='Bz')
    ax2.legend()
    ax2.set_xlabel('UT', fontsize=12)
    ax2.set_ylabel('B', fontsize=12)
    ax2.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])

    # Saves a list of vertical horizontal coordinates and vertical objects
    x_coords = []
    lines1 = []
    lines2 = []

    # A flag variable that controls whether a vertical bar can be inserted
    can_insert_line = True

    # A function that handles click events
    def on_click(event):
        nonlocal can_insert_line
        if event.inaxes and can_insert_line:
            click_x = mdates.num2date(event.xdata)
            click_x = np.datetime64(click_x, 'ms')
            nearest_x = min(mag_mso.Time, key=lambda x_point: abs(x_point - click_x))
            line1 = ax1.axvline(x=nearest_x, color='r', linestyle='--')
            line2 = ax2.axvline(x=nearest_x, color='r', linestyle='--')
            x_coords.append(nearest_x)
            lines1.append(line1)
            lines2.append(line2)
            if len(lines1) == 2:
                can_insert_line = False
            canvas.draw()

    # Bind click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Save the function to CSV file
    def save_to_csv():
        global time_range_index
        file_path = "time_coords.csv"
        new_data = [time_range_index + 1] + x_coords
        if os.path.isfile(file_path):
            # Reading CSV file
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                lines = list(reader)
                # Check whether there is a row corresponding to the current time range index
                for i, line in enumerate(lines):
                    if i > 0 and int(line[0]) == time_range_index + 1:
                        # Replace the data in the current row
                        lines[i] = new_data
                        break
                else:
                    # If there is no row corresponding to the current time_range_index, a new row is added
                    lines.append(new_data)
            # Write the CSV file again
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(lines)
        else:
            # If the file does not exist, write the new data directly
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["number", "X Coordinate 1", "X Coordinate 2"])
                writer.writerow(new_data)
        print(f"Coordinates saved to {file_path}")

    # Function to clear vertical lines
    def clear_lines():
        for line1, line2 in zip(lines1, lines2):
            line1.remove()
            line2.remove()
        lines1.clear()
        lines2.clear()
        x_coords.clear()
        canvas.draw()
        nonlocal can_insert_line
        can_insert_line = True

    # Define amplification function
    def zoom_in():
        if x_coords:
            x_min = min(x_coords)
            x_max = max(x_coords)
            time_range = (mag_mso.Time >= x_min) & (mag_mso.Time <= x_max)
            bt_range = bt[time_range]
            bx_range = mag_mso.Bx[time_range]
            by_range = mag_mso.By[time_range]
            bz_range = mag_mso.Bz[time_range]

            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)
            ax1.set_ylim(min(bt_range), max(bt_range))
            ax2.set_ylim(min(np.concatenate([bx_range, by_range, bz_range])),
                         max(np.concatenate([bx_range, by_range, bz_range])))
            clear_lines()
            canvas.draw()

    # Definition reduction function
    def zoom_out():
        ax1.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])
        ax2.set_xlim(mag_mso.Time[0], mag_mso.Time[-1])
        ax1.set_ylim(min(bt), max(bt))
        ax2.set_ylim(min([min(mag_mso.Bx), min(mag_mso.By), min(mag_mso.Bz)]),
                     max([max(mag_mso.Bx), max(mag_mso.By), max(mag_mso.Bz)]))
        nonlocal can_insert_line
        can_insert_line = True
        canvas.draw()

    # Create button frame
    button_frame = ttk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

    # Create button
    clear_button = ttk.Button(button_frame, text="Clear Lines", command=clear_lines)
    clear_button.grid(row=0, column=0, padx=10, pady=10)

    save_button = ttk.Button(button_frame, text="Save to CSV", command=save_to_csv)
    save_button.grid(row=0, column=1, padx=10, pady=10)

    zoom_button = ttk.Button(button_frame, text="Zoom In", command=zoom_in)
    zoom_button.grid(row=0, column=2, padx=10, pady=10)

    zoom_out_button = ttk.Button(button_frame, text="Zoom Out", command=zoom_out)
    zoom_out_button.grid(row=0, column=3, padx=10, pady=10)

    # Define the Last button function
    def last_plot():
        global time_range_index
        if time_range_index > 0:
            time_range_index -= 1
            root.destroy()
            MAG_GUI(time_ranges)

    # Define the Next button function
    def next_plot():
        global time_range_index
        if time_range_index < len(time_ranges) - 1:
            time_range_index += 1
            root.destroy()
            MAG_GUI(time_ranges)

    next_button = ttk.Button(button_frame, text="Next", command=next_plot)
    next_button.grid(row=0, column=4, padx=10, pady=10)
    last_button = ttk.Button(button_frame, text="Last", command=last_plot)
    last_button.grid(row=0, column=5, padx=10, pady=10)
    # Label information
    marked_label = ttk.Label(button_frame, text="")
    marked_label.grid(row=0, column=6, padx=10, pady=10)

    # Check if it is marked
    def check_marked():
        global time_range_index
        file_path = "time_coords.csv"
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:  # 检查文件是否存在且不为空
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # 跳过头行
                lines = list(reader)
                marked = False
                for line in lines:
                    if len(line) > 0 and int(line[0]) == time_range_index + 1:
                        marked = True
                        break
                if marked:
                    marked_label.config(text="Marked: Yes")
                else:
                    marked_label.config(text="Marked: No")
        else:
            marked_label.config(text="Marked: No")

    check_marked()
    root.mainloop()
    root.destroy()

def Download_MESSENGER_Mag_Data(year, doy, mode):
    # Download MESSENGER_MAG_DATA,
    Month = int(MESSENGER_MAG.doy_to_date(year, doy)[5:7])
    # Get the first day of the month
    first_day = datetime(year, Month, 1)
    # Get the first day of the next month
    if Month == 12:
        next_month_first_day = datetime(year + 1, 1, 1)
    else:
        next_month_first_day = datetime(year, Month + 1, 1)
    # Get the last day of the month
    last_day = next_month_first_day - timedelta(days=1)
    # Calculate the first day of the month and the last day of the year
    first_day_of_year = datetime(year, 1, 1)
    first_day_of_year_number = (first_day - first_day_of_year).days + 1
    last_day_of_year_number = (last_day - first_day_of_year).days + 1
    first_last_month_str = str(first_day_of_year_number).zfill(3) + '_' + str(last_day_of_year_number).zfill(3) + '_'
    Month_str = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    # # Get download link
    if mode == '0.05':
        URL = 'https://search-pdsppi.igpp.ucla.edu/ditdos/download?id=pds://PPI/mess-mag-calibrated/data/mso/' + str(
            year) + '/' + first_last_month_str + Month_str[Month - 1] + '/'
        tabfilename = 'MAGMSOSCI' + str(year)[2:4] + str(doy).zfill(3) + '_V08.TAB'
        download_dir = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Science_MAG/TAB/'
    else:
        URL = 'https://search-pdsppi.igpp.ucla.edu/ditdos/download?id=pds://PPI/mess-mag-calibrated/data/mso-avg/' + str(
            year) + '/' + first_last_month_str + Month_str[Month - 1] + '/'
        tabfilename = 'MAGMSOSCIAVG' + str(year)[2:4] + str(doy).zfill(3) + '_' + mode + '_V08.TAB'
        download_dir = MESSENGER_data_folder + '/MESSENGER_Data/MAG/Reduced_MAG/TAB/'
    # Download and save to local
    download_url = URL + tabfilename
    urllib.request.urlretrieve(download_url, download_dir + tabfilename)

def Read_Sun_MP_BS(Sun_csv):
    # Reading CSV file
    df_Sun = pd.read_csv(Sun_csv)

    # Initializes the list of counters and eligible subscripts
    mp_indices = []
    bs_indices = []

    # Iterate over the DataFrame, recording eligible subscripts
    for i in range(len(df_Sun)):
        if df_Sun['Type'][i][0:2] == 'mp':
            mp_indices.append(i)
        else:
            bs_indices.append(i)

    # Defines the dtype of the array of records
    dtype = [('start', 'object'),
             ('end', 'object'),
             ('type', 'object')]

    # Creates an empty array of records
    mplist = np.recarray(len(mp_indices), dtype=dtype)
    bslist = np.recarray(len(bs_indices), dtype=dtype)
    print(df_Sun['start'][0])
    # Populate the array of records with subscripts
    for idx, i in enumerate(mp_indices):
        mplist[idx] = (df_Sun['start'][i], df_Sun['end'][i], df_Sun['Type'][i])

    for idx, i in enumerate(bs_indices):
        bslist[idx] = (df_Sun['start'][i], df_Sun['end'][i], df_Sun['Type'][i])

    # Save as an npy file
    np.save(os.path.join(MESSENGER_data_folder, 'mplist.npy'), mplist)
    np.save(os.path.join(MESSENGER_data_folder, 'bslist.npy'), bslist)

MAG_GUI(time_list='BS_Sun')







