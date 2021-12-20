# JavaScript Snippets

# Table of Contents

- [Commands](#-commands)
- [Template Strings/Literals](#-template-strings--literals)
- [Arrays](#-arrays)
- [If/Else Statements](#-ifelse-statements)
  - [Ternary Operators](#ternary-operators)
  - [Switch Statements](#switch-statements)
- [Loops](#-loops)
- [Functions](#-functions)
  - [Function Declaration](#function-declaration)
  - [Function Expression / Anoymous](#function-expression--anoymous-function)
  - [Arrow Function](#arrow-function)
- [Objects](#-objects)

---

Javascript Datatypes : `Number`, `String`, `Boolean`, `Undefined`, `Null`.

Falsy Values : `0`, `''`, `undefined`, `null`, `NaN`

JavaScript allows type coercion; or converting dtype of value as needed

code snippet added to html file to import the `src` javascript file.

```javascript
<script type="text/javascript" src="app.js"></script>
// src is "source" of js **file**
```

# Commands

---

|                 Command                  |                                                                                 |
| :--------------------------------------: | ------------------------------------------------------------------------------- |
|             `'use strict';`              | turns on strict mode, which doesn't allow silent errors; insert at top of file. |
| `document.querySelector('html-element')` | accesses the HTML element passed to allow for DOM manipulation.                 |
|      `console.log('Hello World!')`       | displays output (text, variable, etc.) to the browser's console log             |
|         `alertF('Hello World!')`         | displays browser alert with output (text, variable, etc.)                       |
|    `prompt("What is todays' date?")`     | displays browser prompt for user input; can be stored as a variable             |
|    `console.log(typeof variableName)`    | displays the variable's data type to the browser's console log                  |

datatype conversions for `String` and `Int`

```javascript
// converts the string 1991 into an integer
console.log(Number('1991'));

// converts the integer into a string
console.log(String(1991));
```

# Template Strings / Literals

---

Defines a string with passed values from defined variables using backticks. Newline incorporated automatically.

```javascript
const normalName = 'Peter Parker';
const superheroName = 'Spider-Man';

console.log(`Do you belive ${normalName} could be ${superheroName?`)
```

# Arrays

---

```javascript
// using bracket notation
const friends = ['Iron Man', 'Incredible Hulk', 'Dr. Strange'];

// using Array method
const years = new Array(1991, 1984, 2008, 2020);
```

|          Command          |                                                                            |
| :-----------------------: | -------------------------------------------------------------------------- |
|      `array.length`       | calculates the length of the array specified                               |
| `array.indexOf('Value')`  | returns the index of the element in the array if existing else returns -1  |
| `array.includes('Value')` | returns boolean T/F if the element is in the array                         |
| `array.unshift('Value')`  | adds an element to the beginning of an array; captures new length of array |
|   `array.push('Value')`   | adds an element to the end of an array; captures new length of array       |
|      `array.shift()`      | removes the first element from an array; captures the element removed      |
|       `array.pop()`       | removes the last element from an array; captures the element removed       |

# If/Else Statements

---

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

## Ternary Operators

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

## Switch Statements

Provide logic for if/else comparisons over multiple cases

```javascript
const job = 'photographer';
const criminalsJailed = 100;

// will switch the case to be triggered according to the job value passed through.
switch (job) {
  case 'photographer':
    console.log('I think I found the PERFECT job.');
    break;
  case 'student assistant':
    console.log("Horizon High tuition isn't going to pay itself!!");
    break;
  case 'wrestler':
    console.log('With great power comes great responsibility.');
    break;

  // deafult statements identify the "else" block to trigger
  default:
    console.log('Do superheroes have jobs?');
}

// using TRUE in switch to trigger block of code where conditional is TRUE; useful for range conditionals.
switch (true) {
  case criminalsJailed <= 100:
    console.log('Friendly Neighborhood Spiderman');

  case criminalsJailed > 100 && criminalsJailed <= 500:
    console.log('The Amazing Spiderman');

  case criminalsJailed > 500:
    console.log('The Superior Spiderman');

  default:
    console.log('The Spider Menace');
}
```

# Loops

**_For Loops_** keep running while the condition is TRUE

```javascript
for (let rep = 1; rep <= 10; rep++) {
  // printing 10 reps to the console
  console.log(`Lifting Weights repetition: ${rep} 🏋🏻‍♂️`);
}
```

Loops are used to dynamically read from or create **_arrays._**

```javascript
const breakfastFoods = [
  'Doughnuts',
  'Bagel Sandwich',
  'Breakfast Burrito',
  'Cinnamon Roll',
];
const coffeeToo = [];

for (let i = 0; i < breakfastFoods.length; i++) {
  // reading from an array
  console.log(breakfastFoods[i]);

  // filling a new array
  coffeeToo.push(`${breakfastFoods[i]} & Coffee`);
}
```

# Functions

---

## Function Declaration

**_Function Declarations_** define a generic function; can be called/invoked before being defined.

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
const appleJuice = fruitProcessor(5, 0);
console.log(appleJuice);

// storing different run of fruitProcessor with oranges
const appleOrangeJuice = fruitProcessor(2, 4);
console.log(appleOrangeJuice);
```

## Function Expression / Anoymous Function

**_Anoymous Functions or Function Expressions_** can be stored as a variable; cannot be called/invoked before being defined.

```javascript
// creates an anomyous function, also known as a FUNCTION DECLARATION
const calcAge = function (birthYear) {
  return 2050 - birthYear;
};

const age = calcAge(1990);
console.log(age);
```

## Arrow Function

**_Arrow Functions_** more effectively define a function using arrow notation. Does not have the **_this_** keyword.

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
};

// logging function to the console
console.log(yearsUntilRetirement(1990, 'Chris'));
```

# Objects

---

Building an object using _object literal syntax_

```javascript
// building an object using object literal syntax
const ironMan = {
  firstName: 'Tony',
  lastName: 'Stark',
  job: 'superhero',
  friends: ['Thor', 'Dr. Strange', 'Hulk'],
};

// dot vs. bracket notation
console.log(ironMan.firstName);
console.log(ironMan['lastName']);

// adding new elements to the object by assigning the values in dot/bracket notation
ironMan.location = 'California';
ironMan['twitter'] = '@ironMan2003';
```

## Object Method (Function)

Methods are functions nested directly in an object.

```javascript
const capt_america = {
  name: 'Steve Rogers',
  mass: 92,
  height: 1.95,
  birthYear: 1918,

  calcBMI: function () {
    // this assigns a new element (bmi) to 'this' current object
    this.bmi = this.mass / this.height ** 2;
    return this.bmi;
  },

  calcAge: function () {
    // this assigns a new element (age) to 'this' current object
    this.age = 2021 - this.birthYear;
    return this.age;
  },
};

// calls the methods in object to calculate new elements
console.log(capt_america.calcBMI());
console.log(capt_america.calcAge());
```

# DOM Manipulation

|                   Command                   |                                                                 |
| :-----------------------------------------: | --------------------------------------------------------------- |
|  `document.querySelector('html-element')`   | accesses the HTML element passed to allow for DOM manipulation. |
| `document.querySelectorAll('html-element')` | accesses all HTML elements as `nodes`, similar to arrays.       |

Selecting an element in a `div` or `span` element to update the text

```javascript
let messageDiv = document.querySelector('.message');
messageDiv.textContent = `Updated text for '.message' div.`;
```

Selecting the `body` element to update the background color

```javascript
let bodyDiv = document.querySelector('body');
bodyDiv.style.backgroundColor = 'red';
```

Adds an "event listener" function to decide what actions to preform when a the "check" button is "clicked"

```javascript
document.querySelector('.check').addEventListener('click', function () {
  const message = `You've clicked the check button!`;
  console.log(message);
```

Adds/Removes a class attribute from specified HTML element (ie. show/hide elements on page).

```javascript
let topBanner = document.querySelector('.banner');

// removes hidden class which would then show the banner element
topBanner.classList.remove('hidden');

// adds the hidden class which would then hide the banner element
topBanner.classList.add('hidden');
```
