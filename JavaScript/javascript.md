# JavaScript

---
Javascript Datatypes : `Number`, `String`, `Boolean`, `Undefined`, `Null`.

Falsy Values : `0`, `''`, `undefined`, `null`, `NaN`

JavaScript allows type coercion; or converting dtype of value as needed

code snippet added to html file to import the `src` javascript file.
```javascript
<script type="text/javascript" src="app.js"></script>
// src is "source" of js **file**
```
****
## COMMANDS

|              Command              |                                                                     |
| :-------------------------------: | ------------------------------------------------------------------- |
|          `'use strict';`          | turns on strict mode, which doesn't allow silent errors             |
|   `console.log('Hello World!')`   | displays output (text, variable, etc.) to the browser's console log |
|     `alertF('Hello World!')`      | displays browser alert with output (text, variable, etc.)           |
| `prompt('What is todays' date?')` | displays browser prompt for user input; can be stored as a variable |
|  `console.log(typeof variable)`   | displays the variable's data type to the browser's console log      |

datatype conversions for `String` and `Int`
```javascript
// converts the string 1991 into an integer
console.log(Number('1991'));

// converts the integer into a string
console.log(String(1991));
```

### Template Strings / Literals
Defines a string with passed values from defined variables using backticks. Newline incorporated automatically.
```javascript
const normalName = 'Peter Parker';
const superheroName = 'Spider-Man';

console.log(`Do you belive ${normalName} could be ${superheroName?`)
```

### Arrays
```javascript
// using bracket notation
const friends = ['Iron Man', 'Incredible Hulk', 'Dr. Strange'];

// using Array method
const years = new Array(1991, 1984, 2008, 2020);
```

|          Command          |                                                                            |
| :-----------------------: | -------------------------------------------------------------------------- |
|      `array.length`       | calculates the length of the array specified                               |
| `array.indexOf('Value')`  | returns the index of the element in the array                              |
| `array.includes('Value')` | returns boolean T/F if the element is in the array                         |
| `array.unshift('Value')`  | adds an element to the beginning of an array; captures new length of array |
|   `array.push('Value')`   | adds an element to the end of an array; captures new length of array       |
|      `array.shift()`      | removes the first element from an array; captures the element removed      |
|       `array.pop()`       | removes the last element from an array; captures the element removed       |


### If/Else Statements
`if(conditional) {do this} else {do this instead}` allows code to be ran conditionally. 
```javascript
//  javascript will convert the code passed to the if into a boolean
var lowInk = true;

if (lowInk) {
  console.log(`Order more ink for the printer, it's running low.`);
} else {
  console.log(`You don't have to order more ink for now.`);
}
```

### Ternary Operators
More effectively define an if/else statement; provide shorthand defining same logic
```javascript
const alias = "Peter Parker";

// ternary operators are setups as: conditional ? programming if TRUE : programming if FALSE;
alias === "Peter Parker"
  ? console.log(`I think that dude is Spiderman?`);
  : console.log(`I think his name is ${alias}.`);

//  ternary operators can also be assigned to variables
const criminalsJailed = 100;

// conditional ? programming if TRUE assigned : programming if FALSE assigned;
const ability = criminalsJailed >= 80 ? "novice" : "veteran";

console.log("That superhero is a well known " + ability);
```

### Switch Statements
Provide logic for if/else comparisons over multiple cases
```javascript
const job = "photographer";
const criminalsJailed = 100;

// will switch the case to be triggered according to the job value passed through.
switch (job) {
  case "photographer":
    console.log("I think I found the PERFECT job.");
    break;
  case "student assistant":
    console.log("Horizon High tuition isn't going to pay itself!!");
    break;
  case "wrestler":
    console.log("With great power comes great responsibility.");
    break;

  // deafult statements identify the "else" block to trigger
  default:
    console.log("Do superheroes have jobs?");
}

// using TRUE in switch to trigger block of code where conditional is TRUE; useful for range conditionals.
switch (true) {
  case criminalsJailed <= 100:
    console.log("Friendly Neighborhood Spiderman");

  case criminalsJailed > 100 && criminalsJailed <= 500:
    console.log("The Amazing Spiderman");

  case criminalsJailed > 500:
    console.log("The Superior Spiderman");

  default:
    console.log("The Spider Menace");
}
```

### Functions

***Function Declarations*** define a generic function; can be called/invoked before being defined.
```javascript
// create function to determine makeup of juice from apples/oranges
function fruitProcessor(apples, oranges) {

  // display the count of apples and oranges
  console.log(apples, oranges);

  // storing the makeup of juice string with apple/orange parameters to juice, returning
  const juice = `Juice with ${apples} apples and ${oranges} oranges.`;
  return juice;
}

// storing run of fruitProcessor as appleJuice, print to console
const appleJuice = fruitProcessor(5,0);
console.log(appleJuice);

// storing different run of fruitProcessor with oranges
const appleOrangeJuice = fruitProcessor(2,4);
console.log(appleOrangeJuice);
```

***Anoymous Functions or Function Expressions*** can be stored as a variable; cannot be called/invoked before being defined.
```javascript
// creates an anomyous function, also known as a FUNCTION DECLARATION
const calcAge = function (birthYear) {
  return 2050 - birthYear;
}

const age = calcAge(1990);
console.log(age);
```

### Arrow Function
***Arrow Functions*** more effectively define a function using arrow notation. Does not have the ***this*** keyword. 
```javascript
//one liner functions do not require the 'return' statement
const calcAge = birthYear => 2037 - birthYear;

const age = calcAge(1990);
console.log(age);
```

```javascript
// arrow function with multiple parameters
const yearsUntilRetirement = (birthYear, firstName) => {

    // defining the age and retirement values
    const age = 2037 - birthYear;
    const retirement = 65 - age;

    // returning string with firstName and retirement values
    return `${firstName} retires in ${retirement} years.`;
}

// logging function to the console
console.log(yearsUntilRetirement(1990, 'Chris'));
```

### Objects
Building an object using *object literal syntax*
```javascript
// building an object using object literal syntax
const ironMan = {
  firstName: 'Tony',
  lastName: 'Stark',
  job: 'superhero',
  friends: ['Thor', 'Dr. Strange', 'Hulk']
};

// dot vs. bracket notation
console.log(ironMan.firstName);
console.log(ironMan['lastName']);

// adding new elements to the object by assigning the values in dot/bracket notation
ironMan.location = 'California';
ironMan['twitter'] = '@ironMan2003';
```