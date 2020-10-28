# NATIVE PYTHON

## DEPENDENCIES
|     Command     |                                            |
| :-------------: | ------------------------------------------ |
|  `import csv`   | dependency for manipulating **csv** files  |
|  `import json`  | dependency for manipulating **json** files |
|   `import os`   | depdendency for **os.path.join()** command |
| `import pickle` | dependency for manipulating pickle files   |


## IMPORTING DATA

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

Imports data from similar files using the **glob** library (wildcard search)
```python
import glob

# list stating the differences in the csvs to be imported
csv_list = [1,2,3,4,5]

# initalizing an empty list to store the df's created
df_list = []

# looping over each list_element to create 'file_name'
for list_element in csv_list:
    file_name = f'../Path/To/File/{list_element}_rest_of_filename'

    # using glob to wildcard search the appropriate csv file; file type added at end of string
    for file in glob.glob(file_name + '*.csv'):

        # saving each df generated to 'df_list'
        df_list.append(pd.read_csv(file, index_col=None))

# concatinating the stored dfs together, row-wise or union-style
complete_df = pd.concat(df_list, axis=0)
```

## INSPECTING DATA

Displays the num of items in an object, num of characters in a string, num of elements in an array, etc.
```python
len(object)
```

## MODIFIER COMMANDS

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

Creates a **range object** using specified **start** and **stop** values (stop not inclusive)
```python
# increments are specified using STEP
num_range = range(start, stop, step)

# converts RANGE OBJECT to a list
nums_list = list(num_range)
```

Creates an **enumerate object** with an **index-item pair** for each item in the object specified 
```python
# indexing starting at START value specified
enumerate_values = enumerate(list, start = int)

# converts the ENUMERATE OBJECT to a list
indexed_list = list(enumerate_values)
```

Passes a function using **map** to each item on the list
```python
rounded_values = map(round, list)

# converts the MAP OBJECT to a list
rounded_list = list(rounded_values)

# MAP can be used with lambda to pass anonymous functions without looping
squared_values = map(lambda x: x**2, list)

# converts the MAP OBJECT to a list
squares_list = list(squared_values)
```

Utilizes the **counter module** to create a **counter dict** of each value and their counts
```python
from collections import Counter

# collects the COUNT of values in the list specified
list_count = Counter(list)

# displays a COUNTER DICT of k, v pairs of the value and their counts
print(list_count)
```

Determines all possible **combinations** in a list specified
```python
from itertools import combinations

# '2' indicates NUM OF COMBINATIONS, '*' unpacks the COMBO OBJECT to a list
combos_list = [ *combinations(list, 2) ]
```

Using the **set method** compares the values in two sets (unique values, no duplicates)
```python
# converting two LISTS into SETS; stores unique values (no duplicates)
set_a = set(list_1)
set_b = set(list_2)

# INTERSECTION collects values shared between the two sets
both_sets = set_a.intersection(set_b)

#DIFFERENCE collects values only in the set specified
set_a_only = set_a.difference(set_b)
set_b_only = set_b.difference(set_a)

# SYMMETRIC_DIFFERENCE collects values that exist in ONE of the sets, but NOT BOTH
set_a.symmetric_difference(set_b)

# UNION collects all values from each set (no duplicates)
set_a.union(set_b)

# check to see if a POSSIBLE VALUE is a member of a set; faster than list looping
if 'Possible Value' in set:
	print('Possible Value')
```

## LIST MANIPULATION
|            Method             |                                                                                |
| :---------------------------: | ------------------------------------------------------------------------------ |
|    `list.append('value')`     | adds the value specified to the **end** of list                                |
| `list.insert(index, 'value')` | **inserts** the value into the list at the index specified                     |
|    `list.remove('value')`     | **removes** the element with the corresponding 'value' from the list specified |
|        `list.upper()`         | **uppercases** each string element in the list specified                       |
|        `list.lower()`         | **lowercases** each string element in the list specified                       |
|        `list.title()`         | uppercases the **initial letter** in each element in the list specified        |

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

Creates a **zip object** by combining two lists; the '*' unpacks the zip_object into a list
```python
zipped_list = [*zip(list_1, list2)]

# displays ZIPPED_LIST
print(zipped_list)

# combines lists to the smallest length
```

## LIST COMPREHENSIONS

Creates a new list by applying an expression to each element of an iterable specified
```python
[ expression for element in iterable ]
```

Each element in the iterable is plugged into the expression if the 'if' condition evaluates to true 
```python
[ expression for element in iterable if condition ]
```

**if/else** clause which is used before the 'for loop'
```python
[ x if x in 'aeiou' else '*' for x in 'apple' ]
```

Create a list of uppercase characters from a string
```python
[ s.upper() for s in "Hello World" ]
```



## DICTIONARIES

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

## IF STATEMENTS

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

## FOR LOOPS

Iterates, or 'loops' through each element in the list specified and prints the 'statement'
```python
for element in list:
    print('statement')
```

## NESTED FOR LOOPS 

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

## FUNCTIONS

Functions provide "packaged code";  must be called before code is ran
```python
def function(arg_1, arg_2=42):
	"""Description of what the function does. Google Docstring Style
	Args:
    arg_1 (str): Description of arg_1 that can break onto the next line if needed.
    arg_2 (int, optional): Write optional when an argument has a default value.

    Returns:
    bool: Optional description of the return value Extra lines are not indented.

    Raises:
    ValueError: Include any error types that the function intentionally raises.

    Notes:
    See https://www.datacamp.com/community/tutorials/docstrings-python for more info.
	"""

function()
```

Function that creates a new DataFrame
```python
# Use an immutable variable for the default argument 
def better_add_column(values, df=None):
  """Add a column of `values` to a DataFrame `df`.
    The column will be named "col_<n>" where "n" is
    the numerical index of the column.

  Args:
    values (iterable): The values of the new column
    df (DataFrame, optional): The DataFrame to update.
      If no DataFrame is passed, one is created by default.

  Returns:
    DataFrame
  """

  # Update the function to create a default DataFrame
  if df is None:
    df = pandas.DataFrame()
  
  df['col_{}'.format(len(df.columns))] = values
  return df

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

## CLASSES

Creates class called **Custom_Class**
```python
class Custom_Class():
    
    def __init__(self, attribute_name, attribute_name_two, attribute_name_three):

        self.attribute_name = attribute_name
        self.attribute_name_two = attribute_name_two
        self.attribute_name_three= attribute_name_three
```

**Class Attributes** are variables that are assigned inside classes but outside of **init**
```python
class CsvFile:
 	
    # class attributes
    instances = []
    
    def __init__(self, file):

    	# creates csv file
    	self.data = pd.read_csv(file)
        
        # appends the filename to the INSTANCES class attribute
        self.__class__.instances.append(file)

# CLASS ATTRIBUTES can be accessed via classes, unlike instance methods
CsvFile.instances
```

**Class Methods** allow the first parameter to represent the class, creating multiple instances
```python
class CsvFile:
 	
    # class attributes
    instances = []
    
    def __init__(self, file):
    
    	# creates csv file
    	self.data = pd.read_csv(file)
        
        # appends the filename to the INSTANCES class attribute
        self.__class__.instances.append(file)
    
    @classmethod
    def instantiate(cls, filenames):
    	return map(cls, filenames)
 
 # CsvFile can be called with a csv path defined
csv_1 = CsvFile('document1.csv')
 
 # now can pass multiple csv paths using the INSTANTIATE class method
csv_1, csv_2 = CsvFile.instantiate(['document1.csv', 'document2.csv'])   
```

Creates an 'instance' of 'Class_Name' using the values provided; stored in 'class_instance'
```python
class_instance = Class_Name('value', 'value', 'value')
```

Retrieves information from an 'instance' cooresponding to the 'attribute_name' specified 
```python
class_instance_info = class_instance.attribute_name
```

Showing **Inheritance**, where a class inherits the attribute of a different class
```python
'''
NAME IS A CLASS ATT. OF ANIMAL; INHERITED BY MAMMAL/REPTILE BY PASSING NAME
INTO MAMMAL/REPTILE AS A PARAMETER
'''

# Create a class Animal
class Animal:
	def __init__(self, name):
		self.name = name

# Create a class Mammal, which INHERITS from Animal
class Mammal(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Create a class Reptile, which also inherits from Animal
class Reptile(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Instantiate a mammal with name 'Daisy' and animal_type 'dog': daisy
daisy = Mammal('Daisy', 'dog')

# Instantiate a reptile with name 'Stella' and animal_type 'alligator': stella
stella = Reptile('Stella', 'alligator')

# Print both objects
print(daisy)
print(stella)
```

Showing **Polymorphism**, where a class displays inheritance and also modifies/morphs other attributes.
```python
'''
MAMMAL/REPTILE EACH INHERIT FROM SPINAL_CORD FROM VERTEBRATE
EACH MORPH/MODIFY THE TEMPERATURE_REGULATION CLASS ATTRIBUTE
'''
# create VERTEBRATE class
class Vertebrate:
    spinal_cord = True
    
    def __init__(self, name):
        self.name = name

# create MAMMAL class, inherits spinal_cord from VERTEBRATE 
class Mammal(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = True

# create REPTILE class, inherits spinal_cord from VERTEBRATE
class Reptile(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = False

# Instantiate a mammal with name 'Daisy' and animal_type 'dog'
daisy = Mammal('Daisy', 'dog')

# Instantiate a reptile with name 'Stella' and animal_type 'alligator'
stella = Reptile('Stella', 'alligator')

# Print stella's attributes spinal_cord and temperature_regulation
print("Stella Spinal cord: " + str(stella.spinal_cord))
print("Stella temperature regulation: " + str(stella.temperature_regulation))

# Print daisy's attributes spinal_cord and temperature_regulation
print("Daisy Spinal cord: " + str(daisy.spinal_cord))
print("Daisy temperature regulation: " + str(daisy.temperature_regulation))
```

## CONTEXT MANAGERS
Opens the file and reads the data using the **with context manager**    
```python
with open ('file_name.file_type', 'r') as fileobj:
    print("test")

# r= read data, a= append data, w = write/overwrite, r+= read/write
```

Opens the JSON file specified using the **with context manager**
```python
with open("file_path/file_name.json") as json_file:
    data = json.load(json_file)
```

Opens the pickle file specified using the **with context manager**
```python
with open('pickled_file.pkl', 'w') as file:
    data = pickle.load(file)
```

Writes data to a 'csv' file using the **with context manager**
```python
# 'w' tells the context manager to write to the csvfile
with open(data_output, 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
```

Defining custom context manager that changes the current path to view files, then reverts to original path
```python
import os

@contextlib.contextmanager
def my_context_mgr():

    # save the CURRENT working directory
    old_dir = os.getcwd()

    # change directory to the PATH specified
    os.chdir(path)

    yield

    # change back to OLD_DIR
    os.chir(old_dir)

# running custom CONTEXT MANAGER with path specified
with my_context_mgr('.data_folder/sub_folder'):

    # CONTEXT MANAGER yields the PATH files, then reverts to OLD_DIR
    project_Files = os.listdir()
```
## TIMING AND PROFILING CODE

### TIME MODULE
Utilizes the **time module** to calculate time between **Start** and **End** times
```python
import time

# create START_TIME and END_TIME variables
start_time = time.time()

# run code between START_TIME and END_TIME to calculate RUN_TIME
result = 5 + 2

end_time = time.time()

# difference in seconds is time it took to run, can divide by 60 to get minutes
run_time = end_time - start_time
```

### RUNTIME

Calculates **runtime** with IPython magic command
```python
# magic command is added before the line of code to be analyzed (LINE MAGIC MODE)
%timeit rand_nums = np.random.rand(1000)

# magic command with two %s will analyze the block/cell of code (CELL MAGIC MODE)
%%timeit
nums = []
for x in range(10):
    nums.append(x)

# %timeit runs through the code multiple times to provide and mean and std dev 
# of the actual runtime; also provides the number of runs and loops taken

# FLAGS
'''
-r: sets the number of runs to complete
-n: sets the number of loops to complete
-o: saves the output of %timeit into a variable
'''

times = %timeit -r2 -n10 -o rand_nums = np.random.rand(1000)
```

### RUNTIME PROFILING
**Line Profiler** provides the run times for each line of code in a function with a summary of run times
```python
# loads the line_profiler into the session
%load_ext line_profiler

# lprun runs the profiler on the function specified
%lprun -f function_name function_name(parameter1, parameter2, parameter3)

# FLAGS
'''
-f: indicates a function will be profiled
'''
```

# NUMPY

## DEPENDENCIES
```python
import numpy as np
```

## METHODS
|                  Command                   | What is Calculated                                                                                              |
| :----------------------------------------: | --------------------------------------------------------------------------------------------------------------- |
|              `np.mean(array)`              | The **average** of the array specified                                                                          |
|             `np.median(array)`             | The **median** of the array specified; not as affected by outliers as the **mean**                              |
|              `np.sqrt(value)`              | The **square root** of the value specified                                                                      |
|       `np.power(power_value, value)`       | The **value** specified and raises it by the value of **power_value**                                           |
|              `np.sin(value)`               | **Sine** of the value specified                                                                                 |
|               `np.cov(x, y)`               | The 2D **covariance matrix** for the x and y arrays specified                                                   |
|              `np.var(array)`               | **Variance** (mean squared distance of the data from their mean); units are squared and not useful              |
|              `np.std(array)`               | **Standard deviation** (square root of variance); or spread of the data from the median                         |
|            `np.corrcoef(x, y)`             | **Pearson Correlation coefficient** (condsidered easier to interpret than covariance) for the x and y specified |
| `np.zeroes( num_of_rows , num_of_columns)` | A matrix of **zeroes** of the shape **num_rows** and **num_columns** specified                                  |
|  `np.ones( num_of_rows , num_of_columns)`  | A matrix of **ones** of the shape **num_rows** and **num_columns** specified                                    |
|     `np.percentile(data, [25, 50,75])`     | The **percentiles** of the specified data; the second parameter takes an array of the percentiles requested     |


**Vectorizes** the Pandas series; faster than **apply** for iterating down columns
```python
# vectorizes the values in column specified
df['Column_Name'].values.sum(axis=1)
```

Creates **arr_list**,  a list of arrays.
```python
arr_list = [[1,2,3], [4,5,6], [7,8,9]]
```

Creates a **matrix** from a **arr_list**, a list of arrays. 
```python
my_matrix = np.array(arr_list)
```

Displays the **shape** of the matrix (row and column count).
```python
my_matrix.shape
```

Arranges the values in the array specified; uses a **start** and **end** value, as well as an option **interval** amount. **End** value is not included in calculation.
```python
np.arange( start_int, end_int, interval_int)
```

Returns **linearly spaced** numbers over the specified interval; the **start** and **end** are included in the calculation.
```python
np.linspace( start_int, end_int, amt_of_nums )
```

Creates an **identity matrix** (ones across the diagonal matrix) with the number of rows/columns specified by **matrix_num**.
```python
identity_matrix = np.eye( matrix_num )
```

Seeds the random number generating algorithm to provide **repoduccibility**; provides pseudorandom number generation
```python
np.random.seed(int)
```

Computes a **random number** between 0 and 1; useful for **Bernoulli trials** (experiement with two options, TRUE or FALSE)
```python
# 'size' parameter provides an array with the amount of variables specified
np.random.random(size=10)
```

Computes random values, containing the **num_of_values** of random numbers specified
```python
np.random.rand( num_of_values )
```
 
 Creates an array containing the **num_of_values** of random numbers specified; **randn** samples are from the **standard normal distribution** (0 to 1)
 ```python
 np.random.randn( num_of_values )
 ```

Samples **random rows** and returns records according to the **size** parameter
```python
# returns all columns for the RANDOM_ROWS of size specified
sample_pop = df.iloc[np.randint(low=0, high=df.shape[0], size=int), :]
```


 # PANDAS

 ## DEPENDENCIES

```python
import pandas as pd
```
## IMPORTING DATA

Imports a **csv** file and saves as a DataFrame

```python
data_path = "file_path/file_name.csv"
df = pd.read_csv(data_path)

df.head()
```

Imports a **JSON** file and saves as a DataFrame
```python
url = "https://URL-DIRECTING-TO-JSON-DATA.json"
df = pd.read_json(url, orient='columns')

df.head()
```

Imports a **csv** file and saves as a dictionary
```python
df = pd.read_csv("file_path/file_name.csv")
d = df.to_dict()
```

## EXPORTING DATA

Writes the df to a **csv** file
```python
df.to_csv("file_path/file_name.csv", index=False)

# index=True writes row names (default)
```
Writes the df to an **Excel** file 
```python
df.to_excel("file_path/file_name.xlsx", index=False)

# use 'ExcelWriter' when exporting to multiple spreadsheets is needed
with pd.ExcelWriter('../file_path/file_name.xlsx') as writer:
    
    # stores 'df_one' to an excel sheet with sheet_name specified
    df_one.to_excel(writer, sheet_name='df_one_data', index=False)

    # stores 'df_two' to an excel sheet with sheet_name specified
    df_two.to_excel(writer, sheet_name='df_two_data', index=False)
```

Writes the df to a **JSON file**
```python
df.to_json(file_name)
```

Writes the df in an  **HTML table** format
```python
df.to_html(file_name)
```

Writes the df to the **SQL table** specified
```python
df.to_sql(table_name, connection_object)
```

## INSPECTING DATA
|                      Method                      | What is Displayed                                                                                 |
| :----------------------------------------------: | ------------------------------------------------------------------------------------------------- |
|                   `df.info()`                    | The Index, Datatype, and Memory info                                                              |
|                 `df.describe()`                  | The summary statistics of all possible aggregrate columns in the df (mean, median, average, etc.) |
|                   `df.dtypes`                    | The **data type** of each column in the df (object,float,etc.)                                    |
|                   `df.head(n)`                   | The **first n rows** in the df specified (default n=5)                                            |
|                   `df.tail(n)`                   | The **last n rows** in the df specified (default n=5)                                             |
|                   `df.columns`                   | A list of all **column names** in the df                                                          |
|                   `df.count()`                   | **The total count of variables in each column**; used to identify incomplete / missing rows.      |
|               `df['Column Name']`                | **A pandas Series** off all values from the column specified                                      |
|           `df['Column Name'].unique()`           | Every **unique** value in the column specified                                                    |
|        `df['Column Name'].value_counts()`        | The **counts** of unique values in the column specified                                           |
|     `df["Column Name"] == "String/Var/Int"`      | A Boolean value (True/False) for each row in the column specified                                 |
| `df.sort_values('Column_Name', ascending=False)` | The column in the df specified, **sorted by the values** in the column                            |

## MODIFIER COMMANDS
|                      Method                       |                                                               |
| :-----------------------------------------------: | ------------------------------------------------------------- |
|              `del df["Column Name"]`              | **Deletes** the column specified from the df                  |
|         `df["Column Name"].astype(float)`         | Converts the datatype of the specified column to a **float**  |
|          `df["Column Name"].astype(str)`          | Converts the datatype of the specified column to a **string** |
|         `df["New Column Name"] = [Array]`         | Creates a new column in the df with an list of values         |
| `df["Column Name"].replace("Value", "New Value")` | Replaces a value in the specified column                      |

**Drops** or **deletes** rows with missing information; used to remove incomplete/missing rows; can use other *'how'* parameters
```python
# drops rows from any column with null values
df.dropna(how='any')

# drops rows with null values in a specific column
df['Column Name'].dropna(how='any')
```

**Sets** the df index using one or more existing columns / arrays (of the correct length)
```python
# providing a key array will create a multiindex df
df.set_index(keys, inplace=True)

# 'inplace=True' overwrites the existing df
```

**Resets** the index of the df (also can revert a multi-index df)
```python
df.reset_index(inplace=True)

# 'inplace=True' overwrites the existing df
```

**Renames** the columns specified in the df
```python
df.rename(columns = {
    "Old Name" : "New Name",
    "Old Name Two" : "New Name Two"
})
```

Replaces values using the **replace** method
```python
# replace multiple values with ONE R_VALUE specified
df['Column_Name'].replace(['Value1', 'Value2'], 'R_Value', inplace=True)

# replace multple values with multiple R_VALUES, one-to-one mapping
df['Column_Name'].replace(['Value1', 'Value2', 'Value3'], ['R_Value1', 'R_Value2', 'R_Value3'], inplace=True)

# Replaces multiple values in the specified column using a DICT (fast)
df["Column Name"].replace({"Value1": "New String Value", "Value2": "New String Value"}, inplace=True)

# used for value normalization for a df column
```

**Reorganizes** the df as needed; creates a new object
```python
new_df = df[['Column 2', 'Column 3', 'Column 1']]
```

Creates a **new df**
```python
df = pd.DataFrame({

    "Column Title1": variable,
    "Column Title2": [array],
    "Column Title3": old_df["Column Name"]
    
})
```

Creates a **DataFrame** from a dictonary specified
```python
pd.DataFrame.from_dict(dict_data)
```

Merges **multiple dfs** specified by the shared column specified; similar to **SQL JOIN** method

```python
pd.merge(df_one, df_two, on="Shared Column", how='left')

# 'how' determines left, inner, outer, right join
```

Merges multiple dfs along rows; no shared column is needed
```python
pd.concat([df_one, df_two], axis=1)

# axis=0 (index), axis=1 (columns), default 0
```

**Boolean Filters** help provide logic in filtering and refining data
```python
# produces a boolean series with True/False values for each row in the column specified
bool_mask = df['Column Name'] > 50

# applying 'bool_filter' filters the df to display records where bool_filter is True
df[bool_mask]

# applying ~ before the mask will filter the df to display records where the bool_filter is False
df[~bool_mask]
```

## AGGREGATE COMMANDS
|              Command               | What is Displayed                                                                      |
| :--------------------------------: | -------------------------------------------------------------------------------------- |
|     `df['Column Name'].mean()`     | The average of the values in the column specified                                      |
|     `df['Column Name'].sum()`      | The total of the values in the column specified                                        |
|     `df['Column Name'].min()`      | The lowest value in the column specified                                               |
|     `df['Column Name'].max()`      | The largest value in the column specified                                              |
|    `df['Column Name'].idxmin()`    | The **FIRST** occurence of the index of the **smallest value** in the specified column |
|    `df['Column Name'].idxmax()`    | The **FIRST** occurence of the index of the **largest value** in the specified column  |
| `df.nsmallest(int, 'Column Name')` | The **int amount** of **smallest values** in the column specified                      |
| `df.nlargest(int, 'Column Name')`  | The **int amount** of **largest values** in the column specified                       |



Creates a 'running tally' column summing all numeric values in a particular row (indicated by axis=1)
```python
df['Running Total'] = df.sum(axis=1)
```

Creates a rolling window calculation on the column specified
```python
rolling = df['Column Name'].rolling( window_size_int, min_periods=None)

rolling.sum()

# 'window_size_int' provides the number of observations to be calculated
# sum() can be replaced with count(), mean(), etc.
```

**Bins** the *'data_to_bin'* values based on the *'bins'* increments
```python
data_to_bin = [1,2,3,4,5,6,7,8,9,10]

# can also pass an interval for bins instead (bins = [1-2, 3-4, ...])
bins = 5

# 'bin_labels' are an optional parameter
bin_labels = ['label1', 'label2',...]

# saved as 'bin_data'
bin_data = pd.cut( data_to_bin, bins, labels=bin_labels)
```

Returns a **count** of **Groupby elements** in a format acceptable to be added to the original dataframe as a new column
```python
df['New Column'] = df.groupby('Grouped Column')['Date/Organizing Column'].transform('count')
```

Returns a **rank** of **Groupby elements** in a format acceptable to be added to the original dataframe as a new column
```python
df['New Column'] = df.groupby('Grouped Column')['Date/Organizing Column'].rank(ascending=True, method='first')

# ascending=True will appropriately rank when organized by date
# method='first' will appropriatey rank when a date has multiple records
```

## DATA PARSING COMMANDS

Converts the column specified into a **list**
```python
list_from_column = df['Column Name'].tolist()
```

**Groups** data by values in the column specified; aggregrates by bracketed column and aggregrate method
```python
gropuby_df = df.groupby( ["Column Name1", "Column Name2"] )['Column 3'].count()

# can use sum(), mean(), max(), std(), etc.
```

**Unstacks** a grouped df by more than one column, easier to read format
```python
groupby_df.unstack()
```

**Locates and displays** records according to **row/column indexing**
```python
# uses indexing instead of column or row labels
df.iloc[row_num, col_num]

# retrieves the slice of rows/columns specified
df.iloc[ row:row, col:col ]

# creates a new copy of the slice instead of just referencing the original df
new_df = df.iloc[ row:row, col:col].copy()
```

**Locates and displays** records according to the **row/column labels**
```python
# this example retrieves data from rows 1, 2, and 3 and from the 'Test Scores' column only    
df.loc[['Jan', 'Feb', 'March'] , "Test Scores"]

# slicing with ':' does not require square brackets like an array or list would
df.loc['Jan':'March', "Test Scores"]

# the colon denotes all ROWS to be returned w/specified columns
df.loc[: , ["Column 1", "Column 2", "Column 3"]]

# the colon denotes all COLUMNS to be returned w/specified rows
df.loc[["Row 1", "Row 2", "Row 3"], : ]
```

**Locates and displays** records where the conditional statement is **True**
```python
# displays all columns for rows where the conditional statement is true
df.loc[ df["Column Name"] == "String/Var/Int", :]

# displays all rows for columns where the conditional statement is true
df.loc[ : , df["Column Name"] == "String/Var/Int"]
```

**Iterates** over rows in a df
```python
for index, row in df.iterrows():
    print(row['ColumnName1'], row['ColumnName2'])
```

Applies a **function to a series** in the dataframe and returns the result to a column in the dataframe
```python
# custom function determines if value is 'True' or 'False'
def custom_function(x):
    if x == True:
        return 1
    
    else:
        return 0

# stores the results to a column in the df; axis=0 applies the function row-wise (default), axis=1 applies the function column-wise
df['New Column'] = df['Column Name'].apply(custom_function, axis=0)
```

Applies a **function to a dataframe** and returns the result to a column in the dataframe
```python
# custom function determines if value is 'True' or 'False'; checks multiple columns in the df
def custom_function(df):
    if df['Column One'] == True or df['Column Two'] == True:
        return 1
    
    else:
        return 0

# stores the results to a column in the df; axis=0 applies the function row-wise (default), axis=1 applies the function column-wise
df['New Column'] = df.apply(custom_function, axis=0)
```

Utilizes **transform** to apply a function to the **groupby object** and broadcasts the values
```python
# defining the min-max transformation
min_max_tr = lambda x: (x - x.min()) / (x.max() - x.min())

# creating the GROUPBY_OBJECT
groupby_object = df.groupby('Column_Name')

# apply the transformation to the GROUPBY_OBJECT; applies to all numerical values
transformed_group = groupby_object.transform(min_max_tr)

# defining a LAMBDA function to fill NaN values with the MEDIAN
missing_trans = lambda x: x.fillna(x.median())

# creating a GROUPBY_OBJECT
groupby_object =df.groupby('Column_Name')

# applying the transformation to the GROUPBY_OBJECT
trans_object =groupby_object.transform(missing_trans)
```

## SAMPLING COMMANDS

Samples random rows
```python

# int= # of records to return, axis=0 samples random ROWS
sample_pop = df.sample(int, axis=0) 
```

[**Resample,**](http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) a time-based groupby, provides an aggregration on time series data based on the **rule parameter**, which describes the frequency with which to apply the aggregration function.
```python
# other aggregrates include sum, count, std, etc.
df.resample( rule='A').mean()

# other frequencies/rules include hourly, daily, weekly, monthly, etc
```

## ROLLING COMMANDS

Used to create a **rolling aggregration** (mean, sum, count, std, etc) based on the **window value**; if using daily time stamps, pandas infers the window value as days.
```python
df.rolling(window=int).mean()
```

## EXPANDING COMMANDS

Used to create an **expanding aggregration** (mean, sum, count, std, etc) which takes account all data from the time series up to each point in time.
```python
df.expanding().mean()
```

## ANALYSIS FUNCTIONS
function that consumes an **array of data** to produce the x and y values of an **ECDF** as two arrays to be unpacked
```python
def ecdf(data):
    """
    Compute ECDF for a one-dimensional array of measurements.
    """
    
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

# unpacks the arrays generated from the 'ecdf' function; showing multiple EDCFs computed
x_values, y_values = ecdf(df['Column Name'])
x_values_2, y_values_2 = ecdf(df['Column Name 2'])


# using matplotlib to plot the formatted arrays in an EDCF style; showing multiple EDCFs plotted
plt.plot(x_values, y_values, marker='.', linestyle='none')
plt.plot(x_values_2, y_values_2, marker='.', linestyle='none')

plt.show()
```

function that creates a **Bernoulli Trial**, which returns the number of successes out of n Bernoulli trials
```python
def perform_bernoulli_trials(n, p):
    """
    Perform n Bernoulli trials with success probability p
    and return number of successes.
    """
    
    # Initialize number of successes: n_success
    n_success = 0

    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success
```

# SQLALCHEMY

## IMPORTING DATA

Creating a **connection** to the specified database; displaying table names
```python
# Import CREATE_ENGINE
from sqlalchemy import create_engine

# create an engine connecting to SQLite file
engine = create_engine('sqlite:///database_file.sqlite')

# CONNECTION via created engine
con = engine.connect()

# Print table names
print(engine.table_names())
```

**Reflecting** the database and building the metadata based on that information; opposite
of creating a table by hand.
```python
# MetaData conatains info about reflected table in the database
# Table object reads from the engine, autoloads the columns, and populates the MetaData
from sqlalchemy import create_engine, MetaData, Table

# create an engine connecting to SQLite file
engine = create_engine('sqlite:///database_file.sqlite')

# instantiating a META_DATA object
metadata = MetaData()

# reflecting TABLE_NAME table from the engine
table = Table('table_name', metadata, autoload=True, autoload_with=engine)

# Print TABLE_NAME table MetaData with REPR function
print(repr(table))
```

Connecting to an **existing database**, and creating a **table**
```python

# Import create_engine, MetaData, Table, Column, String, and Integer
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer

# creating an engine to connect to sqllite file
engine = create_engine('sqlite:///filename.sqlite')

# initializing MetaData
metadata = MetaData()

# building the PEOPLE table
census = Table('people', metadata,
               Column('name', String(30)),
               Column('sex', String(1)),
               Column('age', Integer()))

# Create the table in the database
metadata.create_all(engine)
```

## EXPORTING DATA

Converting the ResultSet object **results** to a pandas DataFrame
```python
# Import CREATE_ENGINE
from sqlalchemy import create_engine

# import pandas
import pandas as pd

# create an ENGINE connecting to SQLite file
engine = create_engine('sqlite:///database_file.sqlite')

# CONNECTION via created engine
con = engine.connect()

# building a query for ENTIRE TABLE: 'SELECT * FROM table'
stmt = select([table])

# execute STMT and grab all results
results = connection.execute(stmt).fetchall()

# create df from RESULTS
df = pd.DataFrame(table_results)

# Set COLUMN NAMES as query column names
df.columns = results[0].keys()
```

## FILTERING, ORDERING, AND GROUPING

Using **Group By** function and the **func** function to group and aggregrate query
```python
 # import func
from sqlalchemy import func

# building a query: 'SELECT state, COUNT(age) FROM table...'
stmt = select([table.columns.state, func.count(table.columns.age)])

# building group by: '...GROUP BY(state)'
stmt = stmt.group_by(table.columns.state)

# execute the queryand grapping all results
results = connection.execute(stmt).fetchall()

# print results and keys/column names of the results returned
print(results)
print(results[0].keys())
```

Using **Order By** function to order query in descending order
```python
# Import desc
from sqlalchemy import desc

# building a query: 'SELECT state FROM table...'
stmt = select([table.columns.state])

# building order by: '...ORDER BY state DESC'
rev_stmt = stmt.order_by(desc(table.columns.state))

# execute the query and grabbing all results
rev_results = con.execute(rev_stmt).fetchall()

# Print the first 10 rev_results
print(rev_results[:10])
```

Counting the **distinct values** from the column specified, and displaying the **scalar value**
```python
# building a query: 'SELECT COUNT(DISTINCT(state)) FROM table...'
stmt = select([func.count(table.columns.state.distinct())])

# execute the query and grab the SCALAR result
distinct_state_count = connection.execute(stmt).scalar()

# Print the distinct_state_count
print(distinct_state_count)
```

## ADVANCED QUERIES

Building a query to calculate a value between two columns; group by **state** and order by **pop change**. Limiting query to 5 records.
```python
# buildng the query: 'SELECT...'
stmt = select([

    # selecting STATE column..
    table.columns.state, 

    # calculating difference between in POPULATION from 2000 to 2008 and labeling as POP_CHANGE
    (table.columns.pop2008 - table.columns.pop2000).label('pop_change')

    ])

# buidling group by: '...GROUP BY state'
stmt_grouped = stmt.group_by(census.columns.state)

# building order by: '...ORDER BY 'pop_change' DESC
stmt_ordered = stmt_grouped.order_by(desc('pop_change'))

# buildling limit: '...LIMIT(5)'
stmt_top5 = stmt_ordered.limit(5)

# execute the query and grab 5 RECORDS
results = connection.execute(stmt_top5).fetchall()
```

## MODIFYING DATA

**Inserting** a row into an existing table; using **select** statement to verify updates
```python
# Import insert and select from sqlalchemy
from sqlalchemy import insert, select

# building an INSERT statement to insert a record into the table 
insert_stmt = insert(table).values(name='Chris', count=1, amount=100.00, valid=True)

# execute the INSERT statement 
results = con.execute(insert_stmt)


# build a SELECT statement to validate the record was added
proof_stmt = select([table]).where(data.columns.name == 'Chris')

# executing the SELECT statement and printing results
proof_result = con.execute(proof_stmt).first()
print(proof_result)
```

Inserting multiple rows into an existing table by building a list of dictionaries.
```python

# Build a list of dictionaries: values_list
# building a LIST of DICTS
values_list = [
    {'name': 'Chris', 'count': 1, 'amount': 100.00, 'valid': True},
    {'name': 'Nevi', 'count':2, 'amount':75.00, 'valid':False}
]

# building an INSERT statement to insert the records
stmt = insert(table)

# executing the INSERT statement WITH the VALUES_LIST
results = con.execute(stmt, values_list)

# display the ROWCOUNT of the table to confirm updates.
print(results.rowcount)
```

Using **pandas** to insert multiple rows into an existing table
```python
# import pandas
import pandas as pd

# import the csv as a df
df = pd.read_csv("file_name.csv", header=None)

# APPEND the df to table using if_exists='append'
df.to_sql(name='table', con=con, if_exists='append', index=False)
```

**Deleting** rows from the specified table
```python
# Import delete, select
from sqlalchemy import delete, select

# building DELETE statement
delete_stmt = delete(table).where(table.columns.state == 'Texas')

# execute the DELETE statement and print ROW_COUNT
results = con.execute(delete_stmt)
print(results.rowcount)
```

Deleting rows from the specified table with multiple conditions
```python
# building DELETE statement
delete_stmt = delete(census)

# appending a WHERE clause to target sex and age
delete_stmt = delete_stmt.where(
    and_(census.columns.sex == 'M', census.columns.age == 36))

# execute the DELETE statement
results = connection.execute(delete_stmt)
```

## CREATING TABLES
Creating a **table** using constraints(unique, default, nullable, etc.)
```python
# Import Table, Column, String, Integer, Float, Boolean from sqlalchemy
from sqlalchemy import Table, Column, String, Integer, Float, Boolean

# Define a new table with a name, count, amount, and valid column: data
data = Table('data', metadata,
             Column('name', String(255), unique=True),
             Column('count', Integer(), default=1),
             Column('amount', Float()),
             Column('valid', Boolean(), default=False)
)

# Use the metadata to create the table
metadata.create_all(engine)

# Print the table details
print(repr(metadata.tables['data']))
```

