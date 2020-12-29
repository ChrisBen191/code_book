# JavaScript

---

5 different dtypes in Javascript: Number, String, Boolean, Undefined, Null
JavaScript allows type coercion; or converting dtype of value as needed

script provides the method to import a JS file

```javascript
<script type="text/javascript" src="app.js"></script>
// src is "source" of js **file**
```

## COMMANDS

|              Command              |                                                                            |
| :-------------------------------: | -------------------------------------------------------------------------- |
|           `use strict;`           | turns on strict mode, which doesn't allow silent errors                    |
|   `console.log('Hello World!')`   | displays output (text, variable, etc.) to the browser's console log        |
|     `alertF('Hello World!')`      | displays browser alert with output (text, variable, etc.)                  |
| `prompt('What is todays' date?')` | displays browser prompt for user input; can be stored as a variable        |
|  `console.log(typeof variable)`   | displays the variable's data type to the browser's console log             |
|          `array.length`           | calculates the length of the array specified                               |
|     `array.indexOf('Value')`      | returns the index of the element in the array                              |
|     `array.includes('Value')`     | returns boolean T/F if the element is in the array                         |
|     `array.unshift('Value')`      | adds an element to the beginning of an array; captures new length of array |
|       `array.push('Value')`       | adds an element to the end of an array; captures new length of array       |
|          `array.shift()`          | removes the first element from an array; captures the element removed      |
|           `array.pop()`           | removes the last element from an array; captures the element removed       |


defines a variable
```javascript
var variable_name = "Chris";
```

defines an array
```javascript
// using bracket notation
const friends = ['Iron Man', 'Incredible Hulk', 'Dr. Strange'];
console.log(friends);

// using Array method
const years = new Array(1991, 1984, 2008, 2020);
console.log(years);
```

### If/Else statements
Displays a message to the console depending on the value of the boolean using if/else logic.
```javascript
// creating boolean value isMale
var isMale = true;

if (isMale) {
  console.log(isMale + " , he is male.");
} else {
  console.log(isMale + " , she is female.");
}
```

### Ternary Operators
More effectively define an if/else statement; provide shorthand defining same logic
```javascript
var alias = "Peter Parker";

// ternary operators are setups as: conditional ? programming if TRUE : programming if FALSE;
alias === "Peter Parker"
  ? console.log("I think that dude is Spiderman?")
  : console.log("Their name is " + alias + ".");

//  ternary operators can also be assigned to variables
var criminalsJailed = 100;

// conditional ? programming if TRUE assigned : programming if FALSE assigned;
var ability = criminalsJailed >= 80 ? "novice" : "veteran";

console.log("That superhero is a well known " + ability);
```

### Switch Statements
Provide logic for if/else comparisons over multiple cases
```javascript
var job = "photographer";
var criminalsJailed = 100;

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
Store code that is invoked when the function name is called
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

Creates an ***Anoymous Function or Function Expression*** which can be stored as a variable.
```javascript
// creates an anomyous function, also known as a FUNCTION DECLARATION
const calcAge = function (birthYear) {
  return 2050 - birthYear;
}

const age = calcAge(1990);
console.log(age);
```

### Arrow Function
More effectively defines a function; provides same logic
```javascript
//one liner functions do not require the 'return' statement
const calcAge3 = birthYear => 2037 - birthYear;

const age3 = calcAge3(1990);
console.log(age3);
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