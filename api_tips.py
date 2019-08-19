####################################### DEPENDENCIES  #######################################

import requests
import json

# retrieves saved API key in a 'config.py' file to prevent exposure
from config import API_KEY

####################################### IMPORTING DATA #################################

# stores path to be used for requests
url = "https://URL-WHERE-DATA-IS-LOCATED"

# prints the url; can be copied to browser to view JSON file of data
print(url)

# the request code is executed to ping a website/portal for a 'response' providing data
response = requests.get(url)

# prints the HTTP response status code; 'Response [200]' indicates a connection to url
print(response)

# stores the data in a JSON format
response_json = requests.get(url).json()

################################# INSPECTING DATA #######################################

# 'pretty prints' by indentation specified to organize JSON data received
# 'sort_keys=False' does not sort keys (default)
clean_json = json.dumps(response, indent = 2, sort_keys=True))

# displays the value/object stored in the specified key
clean_json["Key Name"]

# displays the length of values stored in the specified key
len (clean_json["Key Name"])

# 'for loop' used to print each key in the dict
for key in dictionary.keys():
  print(key)

# 'for loop' used to print each value in a dict
for value in dictionary.values():
  print(values)

# 'for loop' used to print each key and value pair in a dict
for k,v in dictionary.items():
  print(k,v)