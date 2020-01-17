# DEPENDENCIES

```python
import pandas as pd
```
# IMPORTING DATA

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

# EXPORTING DATA

Writes the df to a **csv** file
```python
df.to_csv("file_path/file_name.csv", index=False)

# index=True writes row names (default)
```
Writes the df to an **Excel** file 
```python
df.to_excel("file_path/file_name.xlsx", index=False)

# index=True writes row names (default)
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
# INSPECTING DATA
|                      Method                      |                                                                                                                   |
| :----------------------------------------------: | ----------------------------------------------------------------------------------------------------------------- |
|                   `df.info()`                    | Displays Index, Datatype, and Memory info                                                                         |
|                 `df.describe()`                  | Displays summary statistics of all possible aggregrate columns in the df (mean, median, average, etc.)            |
|                   `df.dtypes`                    | Display the **data type** of each column in the df (object,float,etc.)                                            |
|                   `df.head(n)`                   | Displays the **first n rows** in the df specified (default n=5)                                                   |
|                   `df.tail(n)`                   | Displays the **last n rows** in the df specified (default n=5)                                                    |
|                   `df.columns`                   | Displays a list of all **column names** in the df                                                                 |
|                   `df.count()`                   | Displays **total count of variables in each column**; used to identify incomplete / missing rows.                 |
|               `df['Column Name']`                | Displays **all** values of the column specified                                                                   |
|           `df['Column Name'].unique()`           | Displays every **unique** value in the column specified                                                           |
|        `df['Column Name'].value_counts()`        | Displays the **counts** of unique values in the column specified                                                  |
|     `df["Column Name"] == "String/Var/Int"`      | Displays a Boolean value (True/False) for each row in the column specified depending on the conditional statement |
| `df.sort_values('Column_Name', ascending=False)` | Displays the column in the df specified, **sorted by the values** in the column                                   |

# MODIFIER COMMANDS
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

# AGGREGATE COMMANDS
|          Command           |                                                            |
| :------------------------: | ---------------------------------------------------------- |
| `df["Column Name"].mean()` | Displays the average of the values in the column specified |
| `df["Column Name"].sum()`  | Displays the total of the values in the column specified   |
| `df["Column Name"].min()`  | Displays the lowest value in the column specified          |
| `df["Column Name"].max()`  | Displays the largest value in the column specified         |

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

# DATA PARSING COMMANDS

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

Locates and displays records according to **row/column indexing**
```python
# uses indexing instead of column or row labels
df.iloc[row_num, col_num]

# retrieves the slice of rows/columns specified
df.iloc[ row:row, col:col ]
```

Locates and displays records according to the **row/column labels**
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

Locates and displays records where the conditional statement is **True**
```python
# displays all columns for rows where the conditional statement is true
df.loc[ df["Column Name"] == "String/Var/Int", :]

# displays all rows for columns where the conditional statement is true
df.loc[ : , df["Column Name"] == "String/Var/Int"]
```

Iterates over rows in a df
```python
for index, row in df.iterrows():
    print(row['ColumnName1'], row['ColumnName2'])
```

# RESAMPLING COMMANDS

[**Resample,**](http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) a time-based groupby, provides an aggregration on time series data based on the **rule parameter**, which describes the frequency with which to apply the aggregration function.
```python
# other aggregrates include sum, count, std, etc.
df.resample( rule='A').mean()

# other frequencies/rules include hourly, daily, weekly, monthly, etc
```

# ROLLING COMMANDS

Used to create a **rolling aggregration** (mean, sum, count, std, etc) based on the **window value**; if using daily time stamps, pandas infers the window value as days.
```python
df.rolling(window=int).mean()
```

# EXPANDING COMMANDS

Used to create an **expanding aggregration** (mean, sum, count, std, etc) which takes account all data from the time series up to each point in time.
```python
df.expanding().mean()
```

# ANALYSIS FUNCTIONS
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