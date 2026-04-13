from numpy import fix,array,log,fmod,arctan,arctan2,arcsin,sqrt,sin,cos,pi,arange
import numpy as np
from datetime import datetime,timedelta
from datetime import date
from typing import List, Union
from numpy import ndarray
import warnings
warnings.filterwarnings("ignore")

def ECEF2geodb(a,b,X,Y,Z):
    '''
    Konverter fra kartesiske ECEF-koordinater til geodetiske koordinater vha Bowrings metode.

    Parameters
    ----------
    a : Store halvakse
    b : Lille halvakse
    X : X-koordinat
    Y : Y-koordinat
    Z : Z-koordinat

    Returns
    -------
    lat : Breddegrad
    lon : Lengdegrad
    h :   Høyde

    '''
    e2m = (a**2 - b**2)/b**2
    e2  = (a**2 - b**2)/a**2
    rho = sqrt(X**2 +Y**2)
    my  = arctan((Z*a)/(rho*b))
    lat = arctan(( Z +e2m*b*(sin(my))**3)/(rho - e2*a*(cos(my))**3))
    lon = arctan2(Y,X)
    N   = Nrad(a,b,lat)
    h   = rho*cos(lat) + Z*sin(lat) - N*( 1 - e2*(sin(lat))**2)
    return lat, lon, h



def ECEF2enu(lat,lon,dX,dY,dZ): ## added this new function 28.01.2023
    """
    Convert from ECEF to a local toposentric coordinate system (ENU)
    """
    ## -- Ensure that longitude is bewtween -180 and 180
    if -2*pi < lon < - pi:
        lon = lon + 2*pi
    elif pi < lon < 2*pi:
        lon = lon - 2*pi

    ## -- Compute sin and cos before putting in to matrix to gain speed
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    dP_ECEF = np.array([[dX, dY, dZ]]).T
    ## -- Propagate through rotation matrix
    M = np.array([[-sin_lon, cos_lon, 0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
        [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]])

    dP_ENU = np.dot(M, dP_ECEF)

    e = dP_ENU[0, 0]
    n = dP_ENU[1, 0]
    u = dP_ENU[2, 0]

    return e, n, u


def ECEF2enu_batch(lat, lon, dX, dY, dZ):
    """
    Vectorized ECEF-to-ENU conversion for a fixed receiver position.

    Parameters
    ----------
    lat : float  – Receiver geodetic latitude (radians)
    lon : float  – Receiver geodetic longitude (radians)
    dX, dY, dZ : numpy arrays – Coordinate differences (satellite − receiver)

    Returns
    -------
    east, north, up : numpy arrays of same shape as dX
    """
    if -2*pi < lon < -pi:
        lon = lon + 2*pi
    elif pi < lon < 2*pi:
        lon = lon - 2*pi

    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    M = np.array([[-sin_lon,           cos_lon,            0],
                  [-sin_lat*cos_lon,   -sin_lat*sin_lon,   cos_lat],
                  [ cos_lat*cos_lon,    cos_lat*sin_lon,   sin_lat]])

    dP = np.array([dX, dY, dZ])   # (3, n)
    enu = M @ dP                   # (3, n)

    return enu[0], enu[1], enu[2]




def Nrad(a,b,lat):
    '''
    Funksjonen beregner Normalkrumningsradiusen for den gitte breddegraden. På engelsk
    "Earth's prime-vertical radius of curvature", eller "The Earth's transverse radius of curvature".
    Den står ortogonalt på M (meridiankrumningsradiusen) for den gitte breddegraden. Dvs øst-vest.

    Parameters
    ----------
    a : Store halvakse
    b : Lille halvakse
    lat : Breddegrad

    Returns
    -------
    N : Normalkrumningsradiusen

    '''

    e2 = (a**2 - b**2)/a**2
    N = a/(1 - e2*sin(lat)**2)**(1/2)
    return N



def compute_satellite_azimut_and_elevation_angle(X, Y, Z, xm, ym, zm):
    """
    Computes the satellites azimute and elevation angle based on satellitte and
    receiver ECEF-coordinates. Utilizes vectorization (no for loops) for
    better performance.

    Unit: Degree.

    Parameters
    ----------
    X : Satellite X-coordinate (numpy array)
    Y : Satellite Y-coordinate (numpy array)
    Z : Satellite Z-coordinate (numpy array)
    xm : Reciever X-coordinate (numpy array)
    ym : Reciever Y-coordinate (numpy array)
    zm : Reciever X-coordinate (numpy array)

    Returns
    -------
    az : Azimut angle in degrees  (numpy array)
    elev: Elevation angle in degrees  (numpy array)


    NOT IN USE AT THE MOMENT
    """

    ## -- WGS 84 ellipsoid:
    a   =  6378137.0         # semi-major ax
    b   =  6356752.314245    # semi minor ax

    # Compute latitude and longitude for the receiver
    lat,lon,h = ECEF2geodb(a,b,xm,ym,zm)

    # Find coordinate difference between satellite and receiver
    dX = (X - xm)
    dY = (Y - ym)
    dZ = (Z - zm)

    # Convert from ECEF to ENU (east,north, up)
    east, north, up = ECEF2enu_batch(lat,lon,dX,dY,dZ)

    # Calculate azimuth angle and correct for quadrants
    azimuth = np.rad2deg(np.arctan(east/north))
    azimuth = np.where((east > 0) & (north < 0) | ((east < 0) & (north < 0)), azimuth + 180, azimuth)
    azimuth = np.where((east < 0) & (north > 0), azimuth + 360, azimuth)

    # Calculate elevation angle
    elevation = np.rad2deg(np.arctan(up / np.sqrt(east**2 + north**2)))
    elevation = np.where((elevation <= 0) | (elevation >= 90), np.nan, elevation) # Set elevation angle to NaN if not in the range (0, 90)

    return azimuth, elevation





def compute_azimut_elev(X,Y,Z,xm,ym,zm):
    """
    Computes the satellites azimute and elevation angel based on satellitte and
    reciever ECEF-coordinates. Can take both single coordinates (float) and list(arrays).

    Unit: Degree.

    Parameters
    ----------
    X : Satellite X-coordinate (float or array)
    Y : Satellite Y-coordinate (float or array)
    Z : Satellite Z-coordinate (float or array)
    xm : Reciever X-coordinate
    ym : Reciever Y-coordinate
    zm : Reciever X-coordinate

    Returns
    -------
    az: Azimut in degrees
    elev: Elevation angel in degrees
    """

    ## -- WGS 84 datumsparametre:
    a   =  6378137.0         # store halvakse
    b   =  6356752.314245    # lille halvakse

    ## -- Beregner bredde og lengdegrad til mottakeren:
    lat,lon,h = ECEF2geodb(a,b,xm,ym,zm)

    ## --Finner vektordifferansen:
    dX = (X - xm)
    dY = (Y - ym)
    dZ = (Z - zm)

    ## -- Transformerer koordinatene over til lokalttoposentrisk system:
    if isinstance(X, float): # if only float put in, not list or array
        east,north,up = ECEF2enu(lat,lon,dX,dY,dZ)
        ## -- Computes the azimut angle and elevation angel for current coordinates (in degrees)
        if (east > 0 and north < 0) or (east < 0 and north < 0):
            az = np.rad2deg(arctan(east/north)) + 180
        elif east < 0 and north > 0:
            az = np.rad2deg(arctan(east/north)) + 360
        else:
            az = np.rad2deg(arctan(east/north))
        # elev = arcsin(up/(sqrt(east**2 + north**2 + up**2)))*(180/pi)
        elev = np.rad2deg(atanc(up, sqrt(east**2 + north**2)))
        if not 0 < elev < 90: # if the satellite is below the horizon (elevation angle is below zero)
            elevation_angle = np.nan
    else:
        east = np.array([]); north = np.array([]); up = np.array([])
        for i in np.arange(0,len(dX)):
            east_,north_,up_ = ECEF2enu(lat,lon,dX[i],dY[i],dZ[i])
            east = np.append(east,east_)
            north = np.append(north,north_)
            up = np.append(up,up_)

        ## -- Computes the azimut angle and elevation angel for list  coordinates (in degrees)
        az = []; elev = []
        for p in np.arange(0,len(dX)):
            # # Kvadrantkorreksjon
            if (east[p]> 0 and north[p]< 0) or (east[p] < 0 and north[p] < 0):
                az.append(np.rad2deg(arctan(east[p]/north[p])) + 180)
            elif east[p] < 0 and north[p] > 0:
                az.append(np.rad2deg(arctan(east[p]/north[p])) + 360)
            else:
                az.append(np.rad2deg(arctan(east[p]/north[p])))
            elev.append(np.rad2deg(atanc(up[p], sqrt(east[p]**2 + north[p]**2))))

    return az,elev


def atanc(y,x):
    z=arctan2(y,x)
    atanc=fmod(2*pi + z, 2*pi)
    return atanc


def date2gpstime(year,month,day,hour,minute,seconds):
    """
    Computing GPS-week nr.(integer) and "time-of-week" from year,month,day,hour,min,sec
    Origin for GPS-time is 06.01.1980 00:00:00 UTC
    """
    t0=date.toordinal(date(1980,1,6))+366
    t1=date.toordinal(date(year,month,day))+366
    week_flt = (t1-t0)/7
    week = fix(week_flt)
    tow_0 = (week_flt-week)*604800
    tow = tow_0 + hour*3600 + minute*60 + seconds

    return week, tow


def glonass_diff_eq(state, acc):
    """
    State is a vector containing x,y,z,vx,vy,vz from navigation message
    """
    J2 = 1.0826257e-3       # Second zonal coefficient of spherical harmonic expression.
    mu = 3.9860044e14       # Gravitational constant [m3/s2]   (product of the mass of the earth and and gravity constant)
    omega = 7.292115e-5     # Earth rotation rate    [rad/sek]
    ae = 6378136.0          # Semi major axis PZ-90   [m]
    r = np.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
    der_state = np.zeros(6)
    if r**2 < 0:
        return der_state
    a = 1.5 * J2 * mu * (ae**2)/ (r**5)
    b = 5 * (state[2]**2) / (r**2)
    c = -mu/(r**3) - a*(1-b)

    der_state[0:3] = state[3:6]
    der_state[3] = (c + omega**2)*state[0] + 2*omega*state[4] + acc[0]
    der_state[4] = (c + omega**2)*state[1] - 2*omega*state[3] + acc[1]
    der_state[5] = (c - 2*a)*state[2] + acc[2]
    return der_state





def gpstime2date(week, tow):
    """
    Calculates date from GPS-week number and "time-of-week" to Gregorian calendar.


    Example:
    week = 2236
    tow = 35898
    date = gpstime2date(week,tow) --> 2022-11-13 09:58:00  (13 november 2022)

    Parameters
    ----------
    week : GPS-week
    tow : "Time of week"

    Returns
    -------
    date : The date given in the Gregorian calender ([year, month, day, hour, min, sec])

    """

    hour = np.floor(tow/3600)
    res = tow/3600 - hour
    min_ = np.floor(res*60)
    res = res*60-min_
    sec = res*60

    # if hours is more than 24, extract days built up from hours
    days_from_hours = np.floor(hour/24)
    # hours left over
    hour = hour - days_from_hours*24

    ## -- Computing number of days
    days_to_start_of_week = week*7

    # Origo of GPS-time: 06/01/1980
    t0 = datetime(1980,1,6)
    t1 = t0 + timedelta(days=(days_to_start_of_week + days_from_hours))

    ## --  Formating the date to "year-month- day"
    t1 = t1.strftime("%Y %m %d")
    t1_ = [int(i) for i in t1.split(" ")]

    [year, month, day] = t1_

    date_ = [year, month, day, hour, min_, sec]
    return date_



def gpstime2date_arrays(week: Union[List[int], np.ndarray], tow: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates date from GPS-week number and "time-of-week" to Gregorian calendar.

    Example:
    -------
    .. code-block:: python

    time_epochs = array([[  2190.        , 518399.99999988],
                         [  2190.        , 518429.99999988],
                         [  2190.        , 531539.99999988],
                         [  2190.        , 531569.99999988]])

    greg_time = gpstime2date_arrays(time_epochs[:,0], time_epochs[:,1])

    Parameters
    ----------
    week : array/list, GPS-week numbers.
    tow  : array/list, "Time of week" values.

    Returns
    -------
    dates : array, An array of dates given in the Gregorian calendar ([year, month, day, hour, min, sec]).
    """
    # Convert week and tow to regular Python lists
    week = week.tolist()
    tow = tow.tolist()
    total_seconds = [w * 7 * 24 * 3600 + t for w, t in zip(week, tow)] # Calculate the time since the GPS epoch (6th January 1980) for each week and tow
    delta = [timedelta(seconds=s) for s in total_seconds] # Calculate the timedelta from the GPS epoch for each week and tow
    gps_epoch = datetime(1980, 1, 6) # The GPS reference epoch
    dates = [gps_epoch + d for d in delta] # Calculate the final date by adding the timedelta to the GPS epoch
    date_array = np.array([[date.year, date.month, date.day, date.hour, date.minute, date.second] for date in dates]) # Create an array by extracting date components

    return date_array


def gpstime2date_arrays_with_microsec(week: Union[List[int], np.ndarray], tow: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates date from GPS-week number and "time-of-week" to Gregorian calendar with microsecond precision.

    Example:
    -------
    .. code-block:: python

    time_epochs = array([[  2190.        , 518399.99999988],
                         [  2190.        , 518429.99999988],
                         [  2190.        , 531539.99999988],
                         [  2190.        , 531569.99999988]])

    greg_time = gpstime2date_arrays(time_epochs[:,0], time_epochs[:,1])

    Parameters
    ----------
    week : array/list, GPS-week numbers.
    tow  : array/list, "Time of week" values.

    Returns
    -------
    dates : array, An array of dates given in the Gregorian calendar ([year, month, day, hour, min, sec, microsec]).
    """
    # Convert week and tow to NumPy arrays for efficient processing
    week = np.asarray(week, dtype=int)
    tow = np.asarray(tow, dtype=float)

    # Calculate the total seconds since GPS epoch
    total_seconds = week * 7 * 24 * 3600 + tow

    # Create a timedelta object for each time in total_seconds
    gps_epoch = datetime(1980, 1, 6)  # The GPS reference epoch
    deltas = [timedelta(seconds=s) for s in total_seconds]

    # Calculate final dates and extract components
    dates = [gps_epoch + delta for delta in deltas]
    date_array = np.array([
        [date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond]
        for date in dates
    ])

    return date_array


def convert_to_datetime_vectorized(time_datetime: np.ndarray) -> list:
    """
    Convert numpy array of datetime components to datetime strings with specified format.
    """
    # Convert datetime components to datetime objects
    datetimes = [datetime(*components) for components in time_datetime]
    # Format datetime objects to strings
    datetime_strings = np.array([dt.strftime("%Y-%m-%d %H:%M:%S") for dt in datetimes]).tolist()
    return datetime_strings


def gpstime_to_utc_datefmt(time_epochs_gpstime: np.ndarray) -> list:
    """
    Coverters form GPS time to UTC with formatting.
    Ex output: "2022-01-01 02:23:30".

    """
    time_datetime = gpstime2date_arrays(*time_epochs_gpstime.T)
    return convert_to_datetime_vectorized(time_datetime)



def date2gpstime_vectorized(gregorian_date_array):
    """
    Convert an array of Gregorian dates to GPS time (week number + time-of-week).

    This is the array version of ``date2gpstime``. Instead of looping over
    each date in Python (via ``np.vectorize``), it computes the result for
    all dates at once using pure NumPy arithmetic.

    The algorithm works in three steps:

    1. **Gregorian → ordinal day number**
       Uses the well-known Julian Day Number formula to convert
       (year, month, day) into a running day count, entirely with integer
       array arithmetic.  The formula first shifts January and February
       into months 11–12 of the *previous* year so that the leap-day
       always falls at the end of the adjusted "year", which makes the
       month-length term ``(153*m + 2) // 5`` work uniformly.

    2. **Ordinal day → GPS week + fractional week**
       Subtracts the GPS epoch (6 Jan 1980) ordinal, then divides by 7
       to get the (fractional) week number.  ``np.fix`` truncates toward
       zero to obtain the integer week.

    3. **Fractional week → time-of-week (TOW)**
       The fractional part of the week is scaled to seconds (×604 800)
       and the intra-day time (hours, minutes, seconds) is added.

    Parameters
    ----------
    gregorian_date_array : array_like, shape (N, 6)
        Each row is ``[year, month, day, hour, minute, second]``.

    Returns
    -------
    weeks : np.ndarray, shape (N,)
        GPS week numbers (rounded to nearest integer).
    tows : np.ndarray, shape (N,)
        Time-of-week in seconds (rounded to nearest integer).

    Examples
    --------
    >>> import numpy as np
    >>> dates = np.array([[2022, 1, 1, 0, 0, 0],
    ...                   [2022, 1, 1, 12, 30, 15]])
    >>> weeks, tows = date2gpstime_vectorized(dates)
    >>> weeks
    array([2190., 2190.])
    >>> tows
    array([518400., 563415.])
    """
    arr = np.asarray(gregorian_date_array)
    years, months, days, hours, minutes, seconds = arr.astype(float).astype(int).T

    # --- Step 1: Gregorian calendar → Julian Day Number (vectorized) ---
    # Shift Jan (month 1) and Feb (month 2) to months 13/14 of the prior
    # year so that the leap day is always the last day of the adjusted year.
    a = (14 - months) // 12            # 1 for Jan/Feb, 0 otherwise
    y = years - a                      # adjusted year
    m = months + 12 * a - 3            # adjusted month (0 = March … 11 = Feb)

    # Julian Day Number from the proleptic Gregorian calendar.
    # The term (153*m + 2)//5 gives the cumulative day count for the
    # adjusted month, exploiting the pattern of 30/31-day months from
    # March onward.
    ordinal = (days
               + (153 * m + 2) // 5   # cumulative days in adjusted months
               + 365 * y              # days from full years
               + y // 4               # leap years every 4 years …
               - y // 100             # … except every 100 years …
               + y // 400             # … but every 400 years
               + 1721119)             # offset to Julian Day epoch

    # --- Step 2: Ordinal → GPS week ---
    # t0: Python ordinal of the GPS epoch (6 Jan 1980), shifted by +366
    #     to match the scalar ``date2gpstime`` convention.
    # t1: Convert our Julian Day Number to the same shifted-ordinal basis
    #     (subtract 1721425, the JDN↔Python-ordinal offset, then add 366).
    t0 = date.toordinal(date(1980, 1, 6)) + 366
    t1 = ordinal - 1721425 + 366
    week_flt = (t1 - t0) / 7.0
    weeks = np.fix(week_flt)           # truncate toward zero → integer week

    # --- Step 3: Fractional week → time-of-week in seconds ---
    tows = ((week_flt - weeks) * 604800.0
            + hours * 3600
            + minutes * 60
            + seconds)

    return np.round(weeks), np.round(tows)



def utc_to_gpst(t_utc):
    """
    Convert from UTC to GPST by adding leap seconds.
    """
    t_gpst = t_utc + get_leap_seconds(t_utc)
    return t_gpst


def get_leap_seconds(week,tow):
    """
    Add leap seconds based on date. Input is week and tow for current obs.
    """
    year,month,day,_,_,_ = gpstime2date(week, tow) # convert to gregorian date
    time = (year,month,day)
    if time <= (2006, 1, 1):
        raise ValueError("Have no history on leap seconds before 2006")
    elif time <= (2009, 1, 1):
        return 14
    elif time <= (2012, 7, 1):
        return 15
    elif time <= (2015, 7, 1):
        return 16
    elif time <= (2017, 1, 1):
        return 17
    else:
        return 18





if __name__=="__main__":
    time_epochs = array([[  2190.        , 518401.40],
                         [  2190.        , 531570.00]])


    date_array = gpstime2date_arrays_with_microsec(time_epochs[:,0], time_epochs[:,1])



