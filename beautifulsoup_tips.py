####################################### DEPENDENCIES  #######################################

import requests

from bs4 import BeautifulSoup

################################## BROWSER SETUP  ##################################

# provides an alias to make requests to websites as a profile and not as a scraper
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

# the request code is executed to ping a website/portal for a 'response' providing data;
# the header alias is included in the request
response = requests.get(url, headers=headers)

# creates a 'soup' object by collecting the text (HTML) from the website using the 'html' parser 
soup = BeautifulSoup( resp.text, "html.parser" )