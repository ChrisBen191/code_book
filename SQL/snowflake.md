# Snowflake <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [Admin Commands](#admin-commands)
- [Metadata Searching](#metadata-searching)
  - [Table Search](#table-search)
  - [Column Search](#column-search)
- [Procedures](#procedures)
  - [Commands](#commands)
  - [JavaScript Framework](#javascript-framework)
  - [SQL Framework](#sql-framework)
- [Tasks](#tasks)
  - [Commands](#commands-1)
  - [Framework](#framework)

## Admin Commands

| Command                                                 |                                              |
| :------------------------------------------------------ | -------------------------------------------- |
| `SHOW GRANTS OF ROLE role_name;`                        | shows access for the `role_name`             |
| `SHOW PARAMETERS FOR WAREHOUSE warehouse_name;`         | shows config parameters for `warehouse_name` |
| `SELECT * FROM DB.INFORMATION_SCHEMA.TABLE_PRIVILEGES;` | Shows privileges for tables in database.     |

## Metadata Searching

### Table Search

```sql
SELECT
    table_name,
    table_type,
    row_count

FROM TEST_DB.INFORMATION_SCHEMA.TABLES

WHERE LOWER(table_name) LIKE '%puppies%'
ORDER BY table_name;
```

### Column Search

```sql
SELECT
    tables_schema,
    table_name,
    column_name

FROM TEST_DB.INFORMATION_SCHEMA.COLUMNS

WHERE LOWER(column_name) LIKE '%dog_breed%'
ORDER BY table_name;
```

## Procedures

### Commands

| Command                                                |                                                       |
| :----------------------------------------------------- | ----------------------------------------------------- |
| `SHOW PROCEDURES IN DB_NAME.schema_name;`              | shows all procedures in a `schema_name`               |
| `DESCRIBE PROCEDURE DB.SCHEMA.procedure_name();`       | provides procedure information for `procedure_name()` |
| `CALL DB.SCHEMA.procedure_name();`                     | calls/runs `procedure_name()`                         |
| `DROP PROCEDURE IF EXISTS DB.SCHEMA.procedure_name();` | drops/deletes the procedure if existing.              |

### JavaScript Framework

- Procedure uses the JavaScript framework to build the procedure; this was the first available procedure format.
- Use `CALL test_db.test_schema.js_test_procedure()` to run the procedure.

```sql
CREATE OR REPLACE PROCEDURE TEST_DB.TEST_SCHEMA.js_test_procedure()
RETURNS VARCHAR
LANGUAGE JAVASCRIPT
COMMENT = 'Place descriptive information on PROCEDURE here.'
AS

$$
try {
    // storing query as variable; use backticks to capture query
    var dataQuery = `
        CREATE OR REPLACE TABLE TEST_DB.TEST_SCHEMA.JS_DATA_TABLE AS (
            SELECT *
            FROM PROD_DB.PROD_SCHEMA.PROD_DATA
            LIMIT 25
        )`;

    // create the query statement and execute by including dataQuery as 'sqlText' value
    var dataQueryStmt = snowflake.createStatement({sqlText: dataQuery});
    var dataRS = dataQueryStmt.execute();

    // store string-text as result if try loop is successful (RETURNS VARCHAR defined in procedure header)
    result = 'Query is completed.';
    }

catch(err)  // if not successful, display information for error
    {
    result =  "Failed: Code: " + err.code + "\n  State: " + err.state;
    result += "\n  Message: " + err.message;
    result += "\nStack Trace:\n" + err.stackTraceTxt;
    }

return result;  // returning either the "success" string or or error string
$$;
```

### SQL Framework

- Procedure uses the SQL framework
- Use `CALL test_db.test_schema.sql_test_procedure()` to run the procedure.

```sql
CREATE OR REPLACE PROCEDURE TEST_DB.TEST_SCHEMA.sql_test_procedure()
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Place descriptive information on PROCEDURE here.'
AS

$$
BEGIN

    CREATE OR REPLACE TABLE TEST_DB.TEST_SCHEMA.TEST_TABLE AS (
        SELECT *
        FROM PROD_DB.PROD_SCHEMA.PROD_DATA
        LIMIT 25
    );

    RETURN 'Query is completed.';

END
$$;
```

- Procedure uses the SQL framework and also allows for a passed parameter.
- Use `CALL test_db.test_schema.sql_test_procedure("10-10-1990")` to run the procedure.
- Snowflake will see `sql_procedure()` and `sql_procedure(send_date)` as different procedures.

```sql
CREATE OR REPLACE PROCEDURE TEST_PROD.TEST_SCHEMA.test_procedure(send_date VARCHAR)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Place descriptive information on PROCEDURE here.'
AS

BEGIN

    CREATE OR REPLACE TABLE TEST_PROD.TEST_SCHEMA.TEST_TABLE AS (
        SELECT *
        FROM PROD_DB.PROD_SCHEMA.PROD_DATA
        WHERE SEND_DATE >= :send_date
        LIMIT 25
    );

    RETURN 'Query is for ' || :send_date || ' is completed.';

END;
```

- Procedure uses multiple passed parameters to create an array of two dates.
- Use `CALL test_db.test_schema.sql_test_procedure("month", -3, "01-01-2022")` to run the procedure.
- Snowflake will see `sql_procedure()`,`sql_procedure(send_date)` and `sql_procedure(date_type, date_increment, report_date)` as different procedures.

```sql
CREATE OR REPLACE PROCEDURE TEST_DB.TEST.SCHEMA.test_procedure(date_type VARCHAR, date_increment NUMBER, report_date VARCHAR)
RETURNS ARRAY
LANGUAGE SQL
COMMENT = 'Creates an array of two dates using passed report_date parameter.'
AS

BEGIN

    LET start_date DATE DEFAULT DATEADD(:date_type, :date_increment, :report_date);
    LET end_date DATE DEFAULT :report_date;

    RETURN array_construct(start_date, end_date);
END;
```

## Tasks

### Commands

| Command                                    |                                                           |
| :----------------------------------------- | --------------------------------------------------------- |
| `SHOW TASKS IN DB..schema_name;`           | shows tasks created in `schema_name`.                     |
| `DESCRIBE TASK DB.SCHEMA.task_name;`       | provides task information for `task_name`.                |
| `EXECUTE TASK DB.SCHEMA.task_name;`        | manually executes/runs `task_name`.                       |
| `ALTER TASK DB.SCHEMA.task_name RESUME;`   | changes task from "suspended" to "active" after creation. |
| `DROP TASK IF EXISTS DB.SCHEMA.task_name;` | drops/deletes task if existing.                           |

### Framework

- Tasks allow the scheduling of a procedure at a regular interval.

```sql
CREATE OR REPLACE TASK TEST_DB.TEST_SCHEMA.test_task
WAREHOUSE = 'WAREHOUSE_NAME'
SCHEDULE = 'USING CRON 00 06 01 * * America/Denver' --scheduled for 6AM on 1st of month, denver time; use crontab.guru
COMMENT = 'Place descriptive information on TASK here.'
AS

-- task is calling/running test_procedure()
CALL TEST_DB.TEST_SCHEMA.test_procedure();

-- once TASK has been created, it automatically enters "suspended" state;
-- running ALTER TASK will activate the TASK if you have PERMISSIONS to do so.
ALTER TASK TEST_DB.TEST_SCHEMA.test_task RESUME;
```
