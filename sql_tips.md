############################## CRUD COMMANDS ##############################
# creates a 'virtual table' saved as a view based on the specified subquery; allows a SELECT query to be saved as a temporary table	
CREATE VIEW view_name AS (subquery);		

# adds a column to the specified table; the column datatype and column definition (NULL or NOT NULL) must be specified	
ALTER TABLE table_name 
ADD COLUMN new_column_name  datatype  column_definition		

# updates an existing column to the specified datatype (int to bigint, varchar(10), etc.)	
ALTER TABLE table_name 
MODIFY column_name  new_data_type		

# inserts data with the values specified, into the columns specified, into the table specified	
INSERT INTO table_name (column_1,  column_2) 
VALUES (value_1,  value_2);		

# changes the specified column name to the new
CHANGE column_name new_column_name		

# drops the specified table from the database
DROP TABLE table_name;	

# drops the specified view from the database
DROP VIEW view_name;	

# deletes the specified column from the table
ALTER TABLE table_name 
DROP column_name;	

############################## QUERY COMMANDS ##############################
# used to connect to the specified database to begin querying
USE database_name;

# displays the specified column(s), use wildcard (*)  
# to query every column in the table specified
SELECT column_name 
FROM table_name;

# assigns an alias to a column/record/table to make easier to query
SELECT column_name AS column_alias 
FROM table_name AS table_alias;	

# returns the number of records from the specified column
SELECT COUNT(column_name) 
FROM table_name;	

# returns all columns  and their dtype from the specified table
SHOW COLUMNS FROM table_name;	

# used with aggregate functions to group the query result by one or more columns; 
# goes before any "ORDER BY" statements	
SELECT column, aggregrated_column 
FROM table_name 
WHERE where_conditions 
GROUP BY non_aggregated_column;		

# used to sort the results in ascending (ASC) or descending (DESC) order; goes 
# after any "GROUP BY" items	
SELECT column_name 
FROM table_name 
ORDER BY column_name;

############################## FILTERING COMMANDS ##############################
# returns the specified 'limit' of records from the query
SELECT column_name 
FROM table_name 
LIMIT variable;	

# returns the specified 'limit' of records from the query after skipping the 'offset'
# number of records
SELECT column_name 
FROM table_name 
LIMIT variable 
OFFSET variable;	

# used to return only distinct (non-duplicated) values from the specified column(s)
SELECT DISTINCT column_name 
FROM table_name;	

# used to filter a any non-aggregate item (date/amount/etc.) from the specified column
SELECT column_name 
FROM table_name 
WHERE column_name = 'variable' 	

# searches for a 'text pattern' in the column specified; use the '%' wildcard for placement, 
# or '_' for a single character query
SELECT column_name 
FROM table_name 
WHERE column_name 
LIKE pattern;	

# used to specify multiple values in a WHERE clause (shorthand for multiple 'OR' conditions), 
# used to establish a subquery
SELECT column_name 
FROM table_name 
WHERE column_name IN 
( value_one, value_two)	

# returns records that are 'between' the values specified; values can be int, text, dates, etc.
SELECT column_name 
FROM table_name 
WHERE column_name
BETWEEN value_1 AND value_2;	

# considered a "WHERE" statement for GROUP BY, allows for filtering on 
# aggregates (COUNT, SUM, MAX,  etc.)
SELECT column_name 
FROM table_name 
GROUP BY column_name 
HAVING condition;	


############################## JOIN COMMANDS ##############################
# returns records that have matching values in both tables; will not produce 'NULL' 
# or unmatched records (may not show all data)
SELECT column_name
FROM table_1 INNER JOIN table_2 
ON table_1.column_name = table_2.column_name; 			

# returns all records from the left table (table_1), and matched records from the right 
# table (table_2); produces 'NULL' on unmatched records	
SELECT column_name 
FROM table_1 RIGHT JOIN table_2 
ON table_1.column_name = table_2.column_name;		

# returns all records from the right table (table_2), and matched records from the left 
# table (table_1); produces 'NULL' on unmatched records	
SELECT column_name
FROM table_1 LEFT JOIN table_2 
ON table_1.column_name = table_2.column_name;		
