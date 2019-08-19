################## USE THE 'API_TIPS.PY' FILE FOR TIPS TO START ############################

##################################### DEPENDENCIES  #######################################

import openweathermapy as own
from config import api_key

####################################### SETTINGS  #######################################

# dict that contains settings to setup responses from openweathermapy
settings = {'units': 'metric', 'appid': api_key}

# retrieves current weather data for the city specified;
# can also be an array of cities
owm.get_current("city", **settings)
