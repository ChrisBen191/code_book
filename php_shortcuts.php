<?php
############################## PHP FOLDER STRUCTURE ##############################
#  file that contains the html which will display or render the data
# 'index.view.php'

# file that contains willthe code to create the data to be rendered in the 
# 'index.view.php' file
# 'index.php'

############################## PHP TERMINAL COMMANDS ##############################

# used in the terminal to provide php options; "help" menu
php -h

# runs a built in server locally (can use any localhost port)
php -S localhost:8888

############################## PHP SHORTCUTS ##############################

# header required at the beginning of every php file
<?php

# footer used to close off the 'php' in a file wtih non-php code; typically 
# not included in a pure php file
?>

# imports or requires data from file specified to be ran.
require 'index.view.php';

# use the '$' to notate a variable 
$variable_name = "Hello World";

# prints the variable or text to the console; must use double-quotes to print
# the variable described, and can use {} with variables to further signify.
echo "Hello World";
echo "Hello " . {$variable_name};

# used to obtain value (variable/key = value) stored in url and printed
echo "Hello, " . $_GET['variable']; 

# when using php in file with non-php code, '<?=' is equivalent to run php
# code, then print the following. 
<?= "Hello, " . $_GET['variable'];

############################## PHP FUNCTIONS ##############################

#function used to 'sanitize' the input to prevent malicious code or urls
htmlspecialchars()