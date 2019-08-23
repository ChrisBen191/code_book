################################ DEPENDENCIES  ################################

# dependency for manipulating json files–
import json

# dependency for manipulating csv files
import csv

# dependency for 'os.path.join()' command
import os
################################ IMPORTING DATA ################################

# creates a list composed of each row in the csv stored as a list; assigned to the 'data' variable
csv_file = open('file.csv')
csv_reader = csv.reader(csv_file)
data = list(csv_reader)

# reads a csv file using the 'os' dependency; 'os.path.join()' requires no back-
# or forwardslashes in the filepath
csv_file = os.path.join("folder_name", "file.csv")

# imports from the JSON file specified
with open("file_path/file_name.json") as json_file:
  data = json.load(json_file)

################################ EXPORTING  DATA ################################

# writes data to a 'csv' file
with open(data_output, 'w', newline="") as csvfile:
  writer = csv.writer(csvfile)
################################ INSPECTING DATA ################################

# displays the num of items in an object, num of characters in a string, num of 
# elements in an array, etc.
len(object)

######################## MODIFIER COMMANDS #########################################

# displays the text below with the variables specified
f'Hello, {variable}. You are {variable_two}.'

# converts the string specified into an integer
int('string')

# converts the string specified into a number with decimal places
float('string')

# converts the number specified into a string
str(1000)

###################################### AGGREGATE COMMANDS  ##################################

# shorthand statement for increasing the 'variable' by 1 (variable = variable + 1); can be used 
#  with all math operators (+, -, /, *)
variable += 1

##################################### LIST MODIFIER COMMANDS  ################################

# adds, or 'appends' the value to the end of the list specified
list.append('value')

# adds the value to the list specified; value is 'inserted' into the list at the index specified 
# and moves every element in the list after the index an addititonal index position
list.insert(index, 'value')

# removes the element with the cooresponding 'value' from the list specified
list.remove('value')

# uppercases each string element in the list specified 
list.upper()

# lowercases each string element in the list specified
list.lower()

# converts the initial capital letter to each element in the list specified
list.title()

# deletes the element with the 'index_num' from the list specified
del list[index_num]

# 'slices' elements from the beginning of the list through 4 (0,1,2,3,4) from the list specified
#  then assigned to 'slice_list'
slice_list = list[:5]

# 'slices' elements 2 through 4 (2,3,4) from the list specified then assigned to 'slice_list'
slice_list = list[2:5]

# 'slices' elements from the 2nd element through the end of the list specified, then assigned to the 'slice_list'
slice_list = list[2:]

##################################### LIST COMPREHENSIONS ################################

# creates a new list by applying an expression to each element of an iterable specified
[ expression for element in iterable ]

# each element in the iterable is plugged into the expression if the 'if' condition 
# evaluates to true 
[ expression for element in iterable if condition ]

# if/else clause which is used before the 'for loop'
[ x if x in 'aeiou' else '*' for x in 'apple' ]

# create a list of uppercase characters from a string
[ s.upper() for s in "Hello World" ]

# opens the file specified and 'reads' the data; to append data to existing file
# use 'a', 'w' to create a file/overwrite an existing file, and 'r+' to read/write
with open ('file_name.file_type', 'r') as fileobj:
  print("test")

##################################### DICTIONARIES ################################

# creates a dictionary using a colon to separate the ' 'keys' and 'values' of the dictionary;
# keys and values can be strings or numbers
dict = {
  'first key': 'value',
  'second key': 'value',
  'third key': 'value'
}

# selects the 'value' with the cooresponding key in the dictionary specified
variable = dict['key']

# selects the 'value' with the cooresponding key in the dictionary specified if the key exists; 
# if not the second parameter is given as the value instead 
variable = dict.get('key',0)

# deletes the key/value pair from the dictionary specified
del dict['key']

# creates a list of dictionaries, using square brackets to encase the dicts; dicts are 
# called by using their 'index' value
dict_list = [
  {'first dict key' : 'value' },
  {'second dict key': 'value' },
  {'third dict key' : 'value' }
]

# selects the 'value' with the cooresponding key in the dictonary specified from a 'dict list'
# using it's 'index' value
variable = dict_list[dict_index]['key']

# appends a new dictionary to the 'dict list' specified
dict_list.append(dict)

# creates a dict of dictionaries 
dict_dict = {
  'dict key' : {'inner dict key':'value'},
  'dict key_2' : {'inner dict key':'value'},
  'dict key_3' : {'inner dict key':'value'}
}

# selects the 'value' with the cooresponding 'inner dict key' in the dictionary specified
#  from a 'dict dict' using it's 'dict key'
variable = dict_dict['dict key']['inner dict key']

# prints 'list value' if it is in the 'list', inside the dictionary specified
if 'list value' in dict[list]:
  print('list value')

# loops through the values of the dictionary specified, and prints the 'value'
for v in dict.values():
  print(v)

# loops through the keys of the dictionary specified, and prints the 'key'
for k in dict.keys():
  print(k)

# loops through the keys and values of the dictionary specified, and prints the 'key/value' pair
for k,v in dict.items():
  print(k,v)

########################################## IF STATEMENTS ################################

# code that prints the 'statement' if the variable specified equals the 'value'; (<, >, <=, >=, !=) can 
# be used in placed of the '==' sign. 'value' can be string, integer, etc.
if variable == 'value':
  print('statement')

# 'if' statement with more than one possible condition; 'elif' is intermediate conditonal
# and 'else' is always used as the final statement (when no conditions were met)
if variable == 'value':
  print('statement')

elif variable == 'value2':
  print('second statement')

else:
  print('third statement')

# 'if' statement when one or more conditions must be met before the 'statment' is printed; if using
# 'or', then at least one of the conditions must be met for the the 'statement' to be printed
if variable == 'value' and variable_two == 'value':
  print('statement')

########################################## FOR LOOPS ################################

# iterates, or 'loops' through each element in the list specified and prints the 'statement'
for element in list:
  print('statement')

############################################### NESTED #############################################

# iterates, or 'loops' through each element in the list specified and then prints the 'statment' if the
# condition is met; 'break' stops looping the code once the condition is met
for element in list:
  
  if element == 'value':
    print('statement')
    break

# elements in 'other_list' (inner loop) are completely iterated over during each iteration of the
# elements in the list specified (outer loop)
for element in list:
  
  for other_element in other_list:
    print(element, other_element)

############################################### FUNCTIONS #############################################

# packaged code under the 'function_name' function; must be called before statement is printed
def function_name():
  print('statement')

function_name()

# packaged code under the 'function_name' function with 'first_number' and 'second_number' arguments;
# when function is called, data is passed into the function (in positional order) and prints the 'total'
def function_name(first_number, second_number):
  total = first_number + second_number
  print(total)

function_name(1 , 3)

# packaged code under the 'function_name' function; when function is called, data is passed into the function
# using the 'keyword' arguments specified (positional order is ignored due to assignment)
def function_name(first_number, second_number):
  total = first_number + second_number
  print(total)

function_name(second_number = 1, first_number = 3)

# packaged code under the 'function_name' function with a default value assigned to the 
# 'second_number' argument; 'default' values do not have to be included when calling the function
# keyword arguments without defaults must come before arguments with an assigned default value  
def function_name(first_number, second_number = 1):
  total = first_number + second_number
  print(total)

function_name(first_number = 3)

# function using the 'return' statement to be able to assign the result of running the function to a variable
def function_name(first_number, second_number):
  total = first_number + second_number
  return total

variable = function_name(1, 3)

############################################### CLASSES #############################################

# creates a 'class' instance called 'Class_Name' with defined attributes specified (def __init__);  
# 'self.attribute...' indicates there will be a variable with the same value as the defined attributes specified
class Class_Name():
  
  def __init__(self, attribute_name, attribute_name_two, attribute_name_three):
    self.attribute_name = attribute_name
    self.attribute_name_two = attribute_name_two
    self.attribute_name_three= attribute_name_three



# creates an 'instance' of 'Class_Name' using the values provided; stored in 'class_instance'
class_instance = Class_Name('value', 'value', 'value')

# retrieves information from an 'instance' cooresponding to the 'attribute_name' specified 
class_instance_info = class_instance.attribute_name