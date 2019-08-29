# DEPENDENCIES

Dependency to allow python to read the system clock for the current time

        import time

Dependency to allow displaying dates in a more convenient format and to allow date arithmetic

        import  datetime

# TIME COMMANDS

Returns the amount of seconds from 01/01/1970 at 12 AM (UTC aka Coordinated Universal Time) as a float value; referred to as an 'epoch timestamp'

        time.time()

Pauses the program the number of 'seconds' specified 

        time.sleep(seconds)

# DATETIME COMMANDS

Converts an 'epoch timestamp' into a 'datetime' object (dt) converted for the local time zone

        dt_converted = datetime.datetime.fromtimestamp(epoch_timestamp)

Returns a 'datetime' object (dt) for the current date and time according to local machine's clock

        dt_current = datetime.datetime.now()

Stores a 'datetime' object (dt) using the integers passed into the datetime parameters assigned to 'dt'

        dt = datetime.datetime(year, month, day, hour, minute, second)

Stores a 'duration' of time specified (year, month, day, etc.), which is then used to preform date arithmetic

        duration = datetime.timedelta(duration)

# PANDAS DATETIME
Creates a range of dates starting with the provided date; # of dates depends on the periods value given, frequency provides time value to iterate (year, month, day, etc).

        pd.date_range( 'YYYY-MM-DD, periods=int, freq='D')

Converts date formats a single format; use optional **format** parameter to map unorthodox date values to make conversion easier.

        pd.to_datetime( [array_of_dates], format='string_code_to_parse')


