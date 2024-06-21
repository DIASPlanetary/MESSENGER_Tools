#File to test code in terminal to get B field time series and trajectories

import MESSENGER_Boundary_ID

if __name__ == "__main__":
    print("Welcome to MUPT terminal!")
    start = input("Enter start time (YYYY-MM-DD-HH-mm-SS): ")
    end = input("Enter Start time (YYYY-MM-DD-HH-mm-SS): ")
    crossings = input("Which crossings do you want to plot? ([None]/philpott/sun/both):")

    if crossings == 'philpott':
        MESSENGER_Boundary_ID.mag_time_series(start,end,philpott=True)
    elif crossings == 'sun':
        MESSENGER_Boundary_ID.mag_time_series(start,end,sun=True)
    elif crossings  == 'both':
        MESSENGER_Boundary_ID.mag_time_series(start,end,sun=True,philpott=True)
    else:
        MESSENGER_Boundary_ID.mag_time_series(start,end)