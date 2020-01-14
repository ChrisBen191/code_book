# DEPENDENCIES
|     Command     |                                            |
| :-------------: | ------------------------------------------ |
|  `import csv`   | dependency for manipulating **csv** files  |
|  `import json`  | dependency for manipulating **json** files |
|   `import os`   | depdendency for **os.path.join()** command |
| `import pickle` | dependency for manipulating pickle files   |


# IMPORTING DATA

Creates a list composed of each row in the csv stored as a list; assigned to the **data** variable
```python
csv_file = open('file.csv')
csv_reader = csv.reader(csv_file)
data = list(csv_reader)
```

Reads a csv file using the *os* dependency; **os.path.join()** requires no back or forward slashes in the filepath
```python
csv_file = os.path.join("folder_name", "file.csv")
```

Imports from the JSON file specified
```python
with open("file_path/file_name.json") as json_file:
    data = json.load(json_file)
```

Imports from the pickle file specified and enters **write** mode
```python
with open('pickled_file.pkl', 'w') as file:
    data = pickle.load(file)
```

# EXPORTING  DATA

Writes data to a 'csv' file
```python
with open(data_output, 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
```


# INSPECTING DATA

Displays the num of items in an object, num of characters in a string, num of elements in an array, etc.
```python
len(object)
```

# MODIFIER COMMANDS

Displays the text below with the variables specified
```python
name = 'Chris'
age = 29

f_string = f'Hello, {name}. You are {age}.'
```

Converts the string specified into an integer, if possible
```python
int('string')
```
Converts the string specified into a number with decimal places, if possible
```python
float('string')
```

Converts the number specified into a string
```python
str(1000)
```

# AGGREGATE COMMANDS

Shorthand statement for increasing the 'variable' by 1 (variable = variable + 1); can be used with all math operators (+, -, /, *)
```python
variable += 1
```

# LIST MANIPULATION
|Method	|   	|
|:---:	|---	|
|`list.append('value')`   	|adds the value specified to the **end** of list
|`list.insert(index, 'value')`	|**inserts** the value into the list at the index specified
|`list.remove('value')`     | **removes** the element with the corresponding 'value' from the list specified
|`list.upper()`  | **uppercases** each string element in the list specified
|`list.lower()` | **lowercases** each string element in the list specified
|`list.title()` | uppercases the **initial letter** in each element in the list specified



Deletes the element with the 'index_num' from the list specified
```python
del list[index_num]
```

*Slices* elements from the beginning of the list through 4 (0,1,2,3,4) from the list specified then assigned to 'slice_list'
```python
slice_list = list[:5]
```

*Slices* elements 2 through 4 (2,3,4) from the list specified then assigned to 'slice_list'
```python
slice_list = list[2:5]
```

*Slices* elements from the 2nd element through the end of the list specified, then assigned to the 'slice_list'
```python
slice_list = list[2:]
```

# LIST COMPREHENSIONS

Creates a new list by applying an expression to each element of an iterable specified
```python
[ expression for element in iterable ]
```

Each element in the iterable is plugged into the expression if the 'if' condition evaluates to true 
```python
[ expression for element in iterable if condition ]
```

*if/else* clause which is used before the 'for loop'
```python
[ x if x in 'aeiou' else '*' for x in 'apple' ]
```

Create a list of uppercase characters from a string
```python
[ s.upper() for s in "Hello World" ]
```

Opens the file specified and reads the data, denoted by **r**; to append data to existing file use **a**. To create a file/overwrite an existing file, use **w**; use **r+** to read/write.
```python
with open ('file_name.file_type', 'r') as fileobj:
    print("test")
```

# DICTIONARIES

Creates a dictionary using a colon to separate the **keys** and **values** of the dictionary; keys and values can be strings or numbers.
```python
dict = {
    'first key': 'value',
    'second key': 'value',
    'third key': 'value'
}
```

Selects the **value** assigned to the denoted key from the dictionary specified
```python
variable = dict['key']
```

Selects the **value** assigned to the denoted key from the dictionary specified; if the key does not exist, the second parameter is given as a value instead.
```python
variable = dict.get('key',0)
```

Deletes the **key/value** pair from the dictionary specified.
```python
del dict['key']
```

Creates **dict_list**, a list of dictionaries, using square brackets to encase the dicts; dicts are called by using their 'index' value
```python
dict_list = [
    {'first dict key' : 'value' },
    {'second dict key': 'value' },
    {'third dict key' : 'value' }
]
```

From **dict_list**, a list of dictionaries, selects the **value** assigned to the denoted key from the dictionary specified by it's index value.
```python
variable = dict_list[dict_index]['key']
```

Appends a new dictionary to **dict_list** specified.
```python
dict_list.append(dict)
```

Creates **dict_dict**, a dictionary of dictionaries, using curly brackets to encase the dicts; dicts are called by using their **dict_key** value. 
```python
dict_dict = {
    'dict key' : {'inner dict key':'value'},
    'dict key_2' : {'inner dict key':'value'},
    'dict key_3' : {'inner dict key':'value'}
}
```

From **dict_dict**, a dictionary of dictionaries, selects the **value** assigned to the denoted **inner_dict_key** from the inner dictionary specified by the **dict_key**. Similar to indexing in list but using key instead.
```python
variable = dict_dict['dict key']['inner dict key']
```

Prints **list_item** if it is in the **list**, inside the dictionary specified
```python
if 'list_item' in dict[list]:
    print('list value')
```

Loops through the **values** of the dictionary specified, and prints the 'value'
```python
for v in dict.values():
    print(v)
```
Loops through the **keys** of the dictionary specified, and prints the 'key'
```python
for k in dict.keys():
    print(k)
```

Loops through the **keys and values** of the dictionary specified, and prints the 'key/value' pair
```python
for k,v in dict.items():
    print(k,v)
```

# IF STATEMENTS

Code that prints the 'statement' if the variable specified equals the 'value'; (<, >, <=, >=, !=) can be used in placed of the '==' sign. 'value' can be string, integer, etc.
```python
if variable == 'value':
    print('statement')
```

**If** statement with more than one possible condition; 'elif' is intermediate conditonal and 'else' is always used as the final statement (when no conditions were met)
```python
if variable == 'value':
    print('statement')

elif variable == 'value2':
    print('second statement')

else:
    print('third statement')
```

**If** statement when one or more conditions must be met before the 'statment' is printed; if using 'or', then at least one of the conditions must be met for the the 'statement' to be printed
```python
if variable == 'value' and variable_two == 'value':
    print('statement')
```

# FOR LOOPS

Iterates, or 'loops' through each element in the list specified and prints the 'statement'
```python
for element in list:
    print('statement')
```

# NESTED FOR LOOPS 

Iterates, or 'loops' through each element in the list specified and then prints the 'statment' if the condition is met; 'break' stops looping the code once the condition is met
```python
for element in list:
    
    if element == 'value':
        print('statement')
        break
```

Elements in 'other_list' (inner loop) are completely iterated over during each iteration of the elements in the list specified (outer loop)
```python
for element in list:
    
    for other_element in other_list:
        print(element, other_element)
```

# FUNCTIONS

Packaged code under the 'function_name' function; must be called before statement is printed
```python
def function_name():
    print('statement')

function_name()
```
Packaged code under the 'function_name' function with 'first_number' and 'second_number' arguments; when function is called, data is passed into the function (in positional order) and prints the 'total'
```python
def function_name(first_number, second_number):

    total = first_number + second_number
    print(total)

function_name(1 , 3)
```

Packaged code under the 'function_name' function; when function is called, data is passed into the function using the 'keyword' arguments specified (positional order is ignored due to assignment)
```python
def function_name(first_number, second_number):

    total = first_number + second_number
    print(total)

function_name(second_number = 1, first_number = 3)
```

Packaged code under the 'function_name' function with a default value assigned to the 'second_number' argument; 'default' values do not have to be included when calling the function keyword arguments without defaults must come before arguments with an assigned default value  
```python
def function_name(first_number, second_number = 1):

    total = first_number + second_number
    print(total)

function_name(first_number = 3)
```

Function using the 'return' statement to be able to assign the result of running the function to a variable
```python
def function_name(first_number, second_number):

    total = first_number + second_number
    return total

variable = function_name(1, 3)
```

# CLASSES

Creates a 'class' instance called 'Class_Name' with defined attributes specified (def __init__);  'self.attribute...' indicates there will be a variable with the same value as the defined attributes specified
```python
class Class_Name():
    
    def __init__(self, attribute_name, attribute_name_two, attribute_name_three):

        self.attribute_name = attribute_name
        self.attribute_name_two = attribute_name_two
        self.attribute_name_three= attribute_name_three
```

Creates an 'instance' of 'Class_Name' using the values provided; stored in 'class_instance'
```python
class_instance = Class_Name('value', 'value', 'value')
```

Retrieves information from an 'instance' cooresponding to the 'attribute_name' specified 
```python
class_instance_info = class_instance.attribute_name
```