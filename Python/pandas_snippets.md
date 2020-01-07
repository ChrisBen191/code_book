# DEPENDENCIES

```python
import pandas as pd
```
# IMPORTING DATA

Imports from a **csv** file

```python
data_path = "file_path/file_name.csv"
df = pd.read_csv(data_path)
df.head()
```
Imports from a **csv** file and converts to a **dict**
```python
df = pd.read_csv("file_path/file_name.csv")
d = df.to_dict()
```
Imports from a **JSON** file
```python
url = "https://URL-DIRECTING-TO-JSON-DATA.json"
df = pd.read_json(url, orient='columns')
df.head()
```
# EXPORTING DATA

Writes the df to a **csv file**; index=True writes row names (default)
```python
df.to_csv("file_path/file_name.csv", index=False)
```
Writes the df to an **Excel file**; index=True writes row names (default)
```python
df.to_excel("file_path/file_name.xlsx", index=False)
```
Writes the df to a **JSON file**
```python
df.to_json(file_name)
```
Saves the df as an **HTML table**
```python
df.to_html(file_name)
```
Writes the df to the **SQL table specified** 
```python
df.to_sql(table_name, connection_object)
```
# INSPECTING DATA

Displays the **first n rows** in the df specified
```python
df.head(n)
```

Displays the **last n rows** in the df specified
```python
df.tail(n)
```

Displays Index, Datatype, and Memory info
```python
df.info()
```

Display the **data type** of each column in the df (object,float,etc.)
```python
df.dtypes
```

Display a list of all **column names** in the df
```python
df.columns
```

Displays **total count of variables in each column**; used to identify incomplete / missing rows.
```python
df.count()
```

Displays the df sorted by the **column specified**, shows the entire df.
```python
df.sort_values('Column Name', ascending=False)
```

Displays summary statistics of all columns in the df (mean, median, average, etc.)
```python
df_description = df.describe()
```

Displays the specified statistic from the column specified; can be assigned to a variable can use count, mean, min, 25%, etc.
```python
std_col_name = df_description['Column Name']['std']
```

Displays the column specified
```python
df["Column Name"] or df.column_name
```
Displays every unique element in the column specified in the df
```python
df["Column Name"].unique()
```
Displays the instances(counts) of unique values in the column specified in the df
```python
df["Column Name"].value_counts()
```
Displays the df sorted by the **column specified** in the df; only shows the column.
```python
df["Column Name"].sort_values(ascending=False) 
```
Displays a Boolean value (True/False) for each row in the column specified depending on the conditional statement
```python
df["Column Name"] == "String/Var/Int"
```

# MODIFIER COMMANDS

Drops rows with missing information; used to remove incomplete / missing rows
```python
df.dropna( how="any")
```

Sets the df index using one or more existing columns / arrays (of the correct length); **inplace = True** does not create a new object, instead it overwrites the existing df
```python
df.set_index(keys, inplace=True)
```

Resets the index of the df (removes multi-index df); **inplace=True** does not create a new object.
```python
df.reset_index(inplace=True)
```

Converts to dtype of the specified column to an integerr
```python
pd.to_numeric(df["Column Name"])
```

Converts the dtype of the specified column to a float; can be used to convert dtype to string (str)
```python
df["Column Name"].astype(float)
```

Deletes the column specified from the df
```python
del df["Column Name"]
```

Renames the columns specified in the df
```python
df.rename(columns = {
    "Old Name" : "New Name",
    "Old Name Two" : "New Name Two"
})
```

Replaces a value in the specified column; used for value normalization for a value in a df column
```python
df.["Column Name"].replace("Value", "New Value" )
```

Replaces the values in the specified column; used for value normalization for values in df column
```python
df["Column Name"].replace({
    "Value1": "New String Value",
    "Value2": "New String Value"
})
```

Reorganizes the df and creates a new object (requires two sets of brackets)
```python
new_df = df[["Column 2","Column 3","Column 1"]]
```

Creates a new df using the array containing the names of the columns to be copied
```python
new_df = df[columns_to_copy_array].copy()
```

Creates a new df that can be assigned to a variable
```python
pd.DataFrame({
    "Column Title1": variable,
    "Column Title2": [array],
    "Column Title3": df["Column Name"]
})
```

Creates a df from a dictonary specified
```python
pd.DataFrame.from_dict(dict_data)
```

Merges two dfs specified by the shared column specified

```python
pd.merge(df_one, df_two, on="Shared Column")
```

Merges two dfs along rows; no shared column is needed
```python
pd.concat([df_one, df_two])
```

Creates 'bins' accoring to the bins array, labeled according to the labels specified
```python
pd.cut( df["Column Name"], bins, labels=group_names)
```

# AGGREGATE COMMANDS

Displays the average of the values in the column specified
```python
df["Column Name"].mean()
```

Displays the total of the values in the column specified
```python
df["Column Name"].sum()
```

Displays the lowest value in the column specified
```python
df["Column Name"].min()
```
Displays the largest value in the column specified
```python
df["Column Name"].max()
```
Creates a new column in the df with an list of values 
```python
df["New Column Name"] = [Array]
```

Creates a 'running tally' column summing summing all numeric values in a particular row (indicated by axis=1)
```python
df['Running Total'] = df.sum(axis=1)
```

Creates a rolling window calculation on the column specified, window_size_int provides the number of observations to be calculated; .sum() can be replaced with count(), mean(), etc.
```python
rolling = df['Column Name'].rolling( window_size_int, min_periods=None,  )
rolling.sum()
```

**Bins** the 'data_to_bin' values based on the 'binsd' increments (can also pass an integer '# of bins' instead); can also pass along optional 'bin_labels'; saved as 'bin_data'.
```python
data_to_bin = [1,2,3,4,5,6,7,8,9,10]
bins = [interval_1, interval_2,...]
bin_labels = ['label1', 'label2',...]
bin_data = pd.cut( data_to_bin, bins, labels=bin_labels)
```

**Qcut** bins the data based on sample quantiles; because sample quantiles are used, the bins will roughly be of equal size. Can also pass own quantiles (#s between 0 and 1, inclusive)
```python
equal_bin_data = pd.qcut(data_to_bin, quartile_cut_integer)
```

# DATA PARSING COMMANDS

Converts the column specified into a list
```python
list_from_column = df['Column Name'].tolist()
```

Displays all rows for the columns specified
```python
df.loc[: , ["Column 1", "Column 2", "Column 3"]]
```

Groups data by values in the column specified (displayed with '.count()'); used to create grouped/binned dfs
```python
df.groupby( ["Column Name1", "Column Name2"] ).count()
```

**Unstacks** a grouped df by more than one column, easier to read formatcd ..
```python
groupby_df.unstack()
```

Displays the mean of values in "Column Name 2" grouped by the values in "Column Name 1"; 'mean' can be replaced with min, max, median, std, count, etc. 
```python
df.groupby("Column Name1")["Column Name2"].mean()
```

Displays rows where the conditional statement is true; used to create specific dfs
```python
df.loc[ df["Column Name"] == "String/Var/Int", :]
```

Allows for more than one conditional statement; can use & (and) or | (or)
```python
df.loc[ (df['Column Name'] == "String/Var/Int" ) & (df['Column Name'] == "String/Var/Int"), : ]
```

Displays the data stored in the df using row/column indexing
```python
df.iloc[row_num, col_num]
```

Displays the data stored in the specified range of rows/columns using row/column indexing
```python
df.iloc[row:row, col:col]
```

Displays the data contained in the specified row and column, must have row/column index
```python
df.loc["Row Name", "Column Name"]
```

Displays the data contained in the rows/columns specified; this method can return duplicates.
```python
df.loc[[ "Row 1", "Row 2"], ["Column X", "Column Y"]]
```

Iterates over rows in a df
```python
for index, row in df.iterrows():
    print(row['ColumnName1'], row['ColumnName2'])
```

# RESAMPLING COMMANDS

**Resample,** a time-based groupby, provides an aggregration (mean, sum, count, std, etc) on time series data based on the **rule parameter**, which describes the frequency with which to apply the aggregration function. [Reference](http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
```python
df.resample( rule='A').mean()
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