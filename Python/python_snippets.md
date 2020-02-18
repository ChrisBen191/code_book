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

## EXPORTING  DATA

Writes data to a 'csv' file
```python
with open(data_output, 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
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

Opens the file specified and reads the data, denoted by **r**; to append data to existing file use **a**. To create a file/overwrite an existing file, use **w**; use **r+** to read/write.
```python
with open ('file_name.file_type', 'r') as fileobj:
    print("test")
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

## CLASSES

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

## TIMING AND PROFILING CODE

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

# RUNTIME PROFILING
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

Function that ranks


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

**Replaces** multiple values in the specified column
```python
df["Column Name"].replace({
    "Value1": "New String Value",
    "Value2": "New String Value"
})

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
bool_filter = df['Column Name'] > 50

# applying 'bool_filter' filters the df to display records where bool_filter is True
df[bool_filter]
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
```

**Locates and displays** records according to the **row/column labels**
```python
# this example retrieves data from rows 1, 2, and 3 and from the 'Test Scores' column only    
df.loc[['Jan', 'Feb', 'March'] , "Test Scores"]

# slicing with ':' does not require square brackets like an array or list would
df.loc['Jan':'March', "Test Scores"]

# the colon denotes all rows to be returned
df.loc[: , ["Column 1", "Column 2", "Column 3"]]

# the colon denotes all columns to be returned
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

## RESAMPLING COMMANDS

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