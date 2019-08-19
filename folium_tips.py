################################## DEPENDENCIES  ##################################

# imports the folium library
import folium

################################## DEPENDENCIES  ##################################

# saves the map to an 'index.html' file; can be accessed in jupyter notebook to 
# open in localhost
m.save('index.html')

################################## MAP CREATION  ##################################

# creates a simple map instance with the coordinates specified DEN(39.739,-104.99)
m = folium.Map(
  location=[lat, lng]
  zoom_start=10
  )

