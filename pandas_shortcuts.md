# DEPENDENCIES

    import pandas as pd

# IMPORTING DATA

Imports from a **csv** file

    data_path = "file_path/file_name.csv"
    df = pd.read_csv(data_path)
    df.head()

Imports from a **csv** file and converts to a **dict**

    df = pd.read_csv("file_path/file_name.csv")
    d = df.to_dict()

Imports from a **JSON** file

    url = "https://URL-DIRECTING-TO-JSON-DATA.json"
    df = pd.read_json(url, orient='columns')
    df.head()

# EXPORTING DATA

Writes the df to a **csv file**; index=True writes row names (default)

    df.to_csv("file_path/file_name.csv", index=False)

Writes the df to an **Excel file**; index=True writes row names (default)

    df.to_excel("file_path/file_name.xlsx", index=False)

Writes the df to a **JSON file**

    df.to_json(file_name)
u
Saves the df as an **HTML table**

    df.to_html(file_name)

Writes the df to the **SQL table specified** 

    df.to_sql(table_name, connection_object)

# INSPECTING DATA

Displays the **first n rows** in the df specified

    df.head(n)

Displays the last n rows in the df specified

    df.tail(n)

Displays Index, Datatype, and Memory info

    df.info()

Display the data type of each column in the df (object,float,etc.)

    df.dtypes

Display a list of all column names in the df

    df.columns

Displays total count of variables in each column; used to identify incomplete / missing rows

    df.count()

Displays the df sorted by the **column specified**, shows the entire df.

    df.sort_values('Column Name', ascending=False)

Displays summary statistics of all columns in the df (mean, median, average, etc.)

    df_description = df.describe()

Displays the specified statistic from the column specified; can be assigned to a variable can use count, mean, min, 25%, etc.

    std_col_name = df_description['Column Name']['std']

Displays the column specified

    df["Column Name"] or df.column_name

Displays every unique element in the column specified in the df

    df["Column Name"].unique()

Displays the instances(counts) of unique values in the column specified in the df

    df["Column Name"].value_counts()

Displays the df sorted by the **column specified** in the df; only shows the column.
    
    df["Column Name"].sort_values(ascending=False) 

Displays a Boolean value (True/False) for each row in the column specified depending on the conditional statement

    df["Column Name"] == "String/Var/Int"

# MODIFIER COMMANDS

Drops rows with missing information; used to remove incomplete / missing rows

    df.dropna( how="any")

Sets the df index using one or more existing columns / arrays (of the correct length); **inplace = True** does not create a new object, instead it overwrites the existing df

    df.set_index(keys, inplace=True)

Resets the index of the df (removes multi-index df); **inplace=True** does not create a new object.

    df.reset_index(inplace=True)

Converts to dtype of the specified column to an integerr

    pd.to_numeric(df["Column Name"])

Converts the dtype of the specified column to a float; can be used to convert dtype to string (str)

    df["Column Name"].astype(float)

Deletes the column specified from the df

    del df["Column Name"]

Renames the columns specified in the df

    df.rename(columns = {
      "Old Name" : "New Name",
      "Old Name Two" : "New Name Two"
    })

Replaces a value in the specified column; used for value normalization for a value in a df column

    df.["Column Name"].replace("Value", "New Value" )

Replaces the values in the specified column; used for value normalization for values in df column

    df["Column Name"].replace({
      "Value1": "New String Value",
      "Value2": "New String Value"
    })

Reorganizes the df and creates a new object (requires two sets of brackets)

    new_df = df[["Column 2","Column 3","Column 1"]]

Creates a new df using the array containing the names of the columns to be copied

    new_df = df[columns_to_copy_array].copy()

Creates a new df that can be assigned to a variable

    pd.DataFrame({
      "Column Title1": variable,
      "Column Title2": [array],
      "Column Title3": df["Column Name"]
    })

Creates a df from a dictonary specified

    pd.DataFrame.from_dict(dict_data)

Merges two dfs specified by the shared column specified

    pd.merge(df_one, df_two, on="Shared Column")

Merges two dfs along rows; no shared column is needed

    pd.concat([df_one, df_two])

Creates 'bins' accoring to the bins array, labeled according to the labels specified

    pd.cut( df["Column Name"], bins, labels=group_names)

# AGGREGATE COMMANDS

Displays the average of the values in the column specified

    df["Column Name"].mean()

Displays the total of the values in the column specified

    df["Column Name"].sum()

Displays the lowest value in the column specified

    df["Column Name"].min()

Displays the largest value in the column specified

    df["Column Name"].max()

Creates a new column in the df with an list of values 

    df["New Column Name"] = [Array]

Creates a 'running tally' column summing summing all numeric values in a particular row (indicated by axis=1)

    df['Running Total'] = df.sum(axis=1)

Creates a rolling window calculation on the column specified, window_size_int provides the number of observations to be calculated; .sum() can be replaced with count(), mean(), etc.

    rolling = df['Column Name'].rolling( window_size_int, min_periods=None,  )
    rolling.sum()

**Bins** the 'data_to_bin' values based on the 'binsd' increments (can also pass an integer '# of bins' instead); can also pass along optional 'bin_labels'; saved as 'bin_data'.

    data_to_bin = [1,2,3,4,5,6,7,8,9,10]
    bins = [interval_1, interval_2,...]
    bin_labels = ['label1', 'label2',...]
    bin_data = pd.cut( data_to_bin, bins, labels=bin_labels)

**Qcut** bins the data based on sample quantiles; because sample quantiles are used, the bins will roughly be of equal size. Can also pass own quantiles (#s between 0 and 1, inclusive)

    equal_bin_data = pd.qcut(data_to_bin, quartile_cut_integer)

# DATA PARSING COMMANDS

Converts the column specified into a list

    list_from_column = df['Column Name'].tolist()

Displays all rows for the columns specified

    df.loc[: , ["Column 1", "Column 2", "Column 3"]]

Groups data by values in the column specified (displayed with '.count()'); used to create grouped/binned dfs

    df.groupby( ["Column Name1", "Column Name2"] ).count()

**Unstacks** a grouped df by more than one column, easier to read formatcd ..
groupby_df.unstack()

Displays the mean of values in "Column Name 2" grouped by the values in "Column Name 1"; 'mean' can be replaced with min, max, median, std, count, etc. 

    df.groupby("Column Name1")["Column Name2"].mean()

Displays rows where the conditional statement is true; used to create specific dfs
    
    df.loc[ df["Column Name"] == "String/Var/Int", :]

Allows for more than one conditional statement; can use & (and) or | (or)

    df.loc[ (df['Column Name'] == "String/Var/Int" ) & (df['Column Name'] == "String/Var/Int"), : ]

Displays the data stored in the df using row/column indexing

    df.iloc[row_num, col_num]

Displays the data stored in the specified range of rows/columns using row/column indexing

    df.iloc[row:row, col:col]

Displays the data contained in the specified row and column, must have row/column index

    df.loc["Row Name", "Column Name"]

Displays the data contained in the rows/columns specified; this method can return duplicates.

    df.loc[[ "Row 1", "Row 2"], ["Column X", "Column Y"]]

Iterates over rows in a df

    for index, row in df.iterrows():
      print(row['ColumnName1'], row['ColumnName2'])

