# Programming with T-SQL

## Creating Variables
Declares a variable that stores values; must start with an @ and its datatype must be specified. Do not use single quotes.

```SQL
DECLARE @send_amount INT, @name CHAR(20), @send_date DATE;

SET @send_amount = 1000
SET @name = chris
SET @send_date = '2019-11-11'
```
## Conditional Processing with Variables

Used to 'make decisions' within programming code; works with 'IF' statement

```SQL
DECLARE @status CHAR(20)
```