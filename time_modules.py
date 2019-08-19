##################################### DEPENDENCIES  ################################

# dependency to allow python to read the system clock for the current time
import time

# dependency to allow displaying dates in a more convenient format and to allow date arithmetic
import  datetime

################################ TIME COMMANDS  ################################

# returns the amount of seconds from 01/01/1970 at 12 AM (UTC aka Coordinated 
# Universal Time) as a float value; referred to as an 'epoch timestamp'
time.time()

# pauses the program the number of 'seconds' specified 
time.sleep(seconds)

################################ DATETIME COMMANDS  ################################

# converts an 'epoch timestamp' into a 'datetime' object (dt) converted for the local time zone
dt_converted = datetime.datetime.fromtimestamp(epoch_timestamp)

# returns a 'datetime' object (dt) for the current date and time according to local machine's clock
dt_current = datetime.datetime.now()

# stores a 'datetime' object (dt) using the integers passed into the datetime parameters; assigned to 'dt'
dt = datetime.datetime(year, month, day, hour, minute, second)

# stores a 'duration' of time specified (year, month, day, etc.), which is then used to preform date arithmetic
duration = datetime.timedelta(duration)

# zeroes out the hour and minute fields in a datetime timestamp to provide clear comparison to non-datetime dates
d
