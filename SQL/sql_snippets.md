# SQL

## CRUD COMMANDS

|               Command                |                                                       |
| :----------------------------------: | ----------------------------------------------------- |
|      `CREATE DATABASE db_name`       | Creates a **database** with the given name specified. |
|       `DROP TABLE table_name`        | **Drops** the specified **table** from the database   |
|        `DROP VIEW view_name`         | **Drops** the specified **view** from the database    |
| `CHANGE column_name new_column_name` | **Changes the column name** to the new name specified |

Allows a table to be **altered** (add column, modify column, etc.)
```SQL
ALTER TABLE table_name

-- Updates an existing column to the specified datatype (int to bigint, varchar(10), etc.)
MODIFY column_name  new_data_type

-- Adds a column to the specified table; the column datatype and column definition (NULL or NOT NULL must be specified
ADD COLUMN new_column_name  datatype  column_definition

-- Deletes the specified column from the table 
DROP column_name
```

**Subsitutes missing data** from the column specified with a value or other column
```SQL
SELECT 
    column_name1,
    ISNULL(column_name2, 'No Value') --can use another column as the replacement parameter
FROM table_name
```

**Subsitutes missing data** from the first non-null column or value specified
```SQL
SELECT
    column_name1,
    COALESCE(column_name2, column_name3, column_name4, 'All Null') --can pass a variable at the end of all columns are null
FROM table_name
```

**Inserts** data with the values specified, into the columns specified, into the table specified	
```SQL
INSERT INTO table_name (column_1,  column_2) 
VALUES (value_1,  value_2)
```

Creates a **virtual table** saved as a view based on the specified subquery; allows a SELECT query to be saved as a temporary table
```SQL
CREATE VIEW view_name AS (subquery)
```

## QUERY COMMANDS

|            Command             |                                                                         |
| :----------------------------: | ----------------------------------------------------------------------- |
|        `SHOW databases`        | Used to display all **databases** that can be connected to              |
|         `SHOW tables`          | Used to display all **tables** from the database connected to           |
|      `USE database_name`       | Used to connect to the **specified database** to begin querying         |
| `SHOW COLUMNS FROM table_name` | Returns **all columns and their dtype** from the specified table        |
| `SELECT DISTINCT column_name`  | Used to return only **non-duplicated** values from the specified column |
|     `LAG(column_name, n)`      | Returns the column's value at the **row n rows before the current row** |
|     `LEAD(column_name, n)`     | Returns the column's value at the **row n rows after the current row**  |
|   `FIRST_VALUE(column_name)`   | Returns the **first value** in the table or partition                   |
|   `LAST_VALUE(column_name)`    | Returns the **last value** in the table or partition                    |


Returns the specified **limit** of records from the query
```SQL
SELECT column_name 
FROM table_name 
LIMIT int
OFFSET int --include OFFSET to skip the 'offset' number of records
```

## AGGREGATE COMMANDS

|                 Command                  |                                                                          |
| :--------------------------------------: | ------------------------------------------------------------------------ |
|                `COUNT(*)`                | Displays **total** number of rows                                        |
|      `COUNT(DISTINCT column_name)`       | Displays **total number of unique values** for the column specified      |
|            `SUM(column_name)`            | Displays the **numeric total** of the values for the column specified    |
|            `ABS(column_name)`            | Displays the **absolute values** (non-negative) for the column specified |
|           `SQRT(column_name)`            | Displays the **square root** values for the column specified             |
|          `SQUARE(column_name)`           | Displays the **squared** values for the column specified                 |
|    `DATEPART(datepart, date_column)`     | Returns the **datepart** ('DD', 'MM', 'YY', etc.) of the date specified  |
| `DATEADD( datepart, value, date_column)` | **Add/Subtract** the value of datepart specified; returns a date         |
| `DATEDIFF(datepart, startdate, enddate)` | Add/Subtract the datepart from the two dates specified; returns a number |

Creates **categories or bins** when the specified conditions are met; stores results in a new column with name specified
```SQL
CASE 
    WHEN BOOLEAN_CONDITION THEN 'a' -- can use AND/OR for multiple conditions
    WHEN BOOLEAN_CONDITION THEN 'b'
    ELSE 'c'
    END AS new_column_name
```

**Counts a record** when the specified conditions are met; stores results in a new column
```SQL
COUNT(CASE
        WHEN BOOLEAN_CONDITION THEN record_id -- Use PK to count rows 
        END) AS new_column_name
```

**Totals the values** from records when the specified conditions are met; stores results in a new column 
```SQL
SUM(CASE
        WHEN BOOLEAN_CONDITION THEN agg_column -- Values will be totaled, can use AVG instead of SUM
        END) AS new_column_name
```

Provides **percentages of records** when the specified conditions are met; stores results in a new column
```SQL
AVG(CASE 
        WHEN BOOLEAN_CONDITION THEN 1
        WHEN BOOLEAN_CONDITION THEN 0
        END) AS pct
```
## WHERE CLAUSE
Filters any **non-aggregate items** (date/amount/etc.) from the specified column
```SQL
SELECT column_name 
FROM table_name 
WHERE column_name = 'variable' 	
```

Searches for a **text pattern** in the column specified
```SQL
SELECT column_name 
FROM table_name 
WHERE column_name 
LIKE pattern --use the '%' wildcard for placement, or '_' for a single character query
```

Used to specify multiple values in a WHERE clause; shorthand for **multiple 'OR' conditions**)
```SQL
SELECT column_name 
FROM table_name 
WHERE column_name IN ( value_one, value_two)	
```

Returns records that are **between** the values specified; values can be int, text, dates, etc.
```SQL
SELECT column_name 
FROM table_name 
WHERE column_name
BETWEEN value_1 AND value_2	
```

## HAVING CLAUSE
Considered a **WHERE statement for GROUP BY**, allows for filtering on aggregates (COUNT, SUM, MAX, etc.)
```SQL
SELECT column_name 
FROM table_name 
GROUP BY column_name 
HAVING condition
```

## JOIN COMMANDS

Returns records that have matching values in **both tables**; will not produce 'NULL' or unmatched records (may not show all data).
```SQL
SELECT column_name
FROM table_1 
INNER JOIN table_2 
    ON table_1.column_name = table_2.column_name; 			
```
Returns all records from the **left table (table_1**), and matched records from the **right table (table_2)**; produces 'NULL' on unmatched records.
```SQL
SELECT column_name 
FROM table_1 
RIGHT JOIN table_2 
    ON table_1.column_name = table_2.column_name;		
```

Returns all records from the **right table (table_2)**, and matched records from the **left table (table_1)**; produces 'NULL' on unmatched records.
```SQL
SELECT column_name
FROM table_1 
LEFT JOIN table_2 
    ON table_1.column_name = table_2.column_name;		
```

# T-SQL

Declares a **variable** that stores a defined value 
```SQL
-- Must start with an @ and its datatype must be specified. Do not use single quotes.
DECLARE @variable_name data_type

--SETS the value for the defined variable
SET @variable_name = 'Value'

-- Displays the defined variable
SELECT @variable_name
```

Creates a **WHILE Loop** that computes code specified within
```SQL
DECLARE @variable_name data_type

SET @variable_name = int

-- Specifies the condition of the WHILE loop
WHILE @variable_name < 10

    BEGIN
        -- Incrementing the value of @variable_name by 1
        SET @variable_name = @variable_name + 1

    END
-- Displays the value of @variable_name once the loop has been broken
SELECT @variable_name
```