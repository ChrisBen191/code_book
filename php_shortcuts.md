
# PHP FOLDER STRUCTURE

File that contains the html which will display or render the data

    index.view.php

File that contains the code to create the data to be rendered in the 'index.view.php' file

    index.php

# PHP TERMINAL COMMANDS

Used in the terminal to provide php options; the "help" menu

    php -h

runs a built in server locally (can use any localhost port)
  
    php -S localhost:8888 

# PHP FILE SETUP

Header required at the beginning of every php file
  
    <?php

Footer used to close off the 'php' in a file wtih non-php code; *typically not included in a pure php file.*
  
    ?>

Imports or requires data from file specified to be ran.

    require 'index.view.php';


# PHP COMMANDS

Use '$' to notate a variable 
  
    $variable_name = "Hello World";

Initializes an empty list
  
    $list_name = [];

Prints the variable or text to the console; must use double-quotes to print the variable described, and can use {} with variables to further signify.

    echo "Hello World";

    echo "Hello " . {$variable_name};

Used to obtain value (variable/key = value) stored in url and printed
    
    echo "Hello, " . $_GET['variable']; 

When using php in file with non-php code, '<?=' is equivalent to run php code, then print the following. 

    <?= "Hello, " . $_GET['variable'];

# PHP FUNCTIONS

Function used to 'sanitize' the input to prevent malicious code or urls

    htmlspecialchars()