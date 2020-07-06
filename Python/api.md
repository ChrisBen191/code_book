# APIs

## DEPENDENCIES
imports the **requests** and **json** dependencies
```python
import requests
import json
```

imports the  **Pretty Print** dependency
```python
from pprint import pprint
```

## COMMANDS
prints the response object to the console
```python
url = 'https://url-to-website.com'

response = request.get(url)
print(response)
```

retrieves the data and converts into **JSON**
```python
url = 'https://url-to-website.com'

# converts response into JSON
response_json = requests.get(url).json()
print(response_json)
```

retrieves the data and converts into **Pretty-Print JSON**
```python
url = 'https://url-to-website.com'

# converts response into JSON
response_json = requests.get(url).json()

# "pretty prints" JSON data by indentation specified
pp_json = json.dumps(response_json, indent=4, sort_keys=True)     
print(pp_json)
```


```python
```
| Command |     |
| :-----: | --- |