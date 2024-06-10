# JavaScript Snippets <!-- omit in toc -->
JavaScript is a high-level, object-oriented, multi-paradigm programming language. JS can be used to create dynamic effects and web applications in the browser, and can also be run outside of web browsers, using `node.js` for the backend.

>In terms of how `HTML`, `CSS`, and `JS` interact, their relationship can be described as analgous to `nouns`, `adjectives`, and `verbs`. `HTML/nouns` describe the elements (ie. paragraph, header), the `CSS/adjective` provides characterization of the element (ie. red paragraph text, round edges), and `JS/verbs` provide actions for the elements (ie. hide paragraph, blur on hover).

# Table of Contents <!-- omit in toc -->

- [Basic Commands](#basic-commands)
- [Operators](#operators)
- [Template Strings / Literals](#template-strings--literals)
- [Arrays](#arrays)
- [Dates](#dates)
- [If/Else Statements](#ifelse-statements)
  - [Ternary Operators](#ternary-operators)
  - [Switch Statements](#switch-statements)
- [Loops](#loops)
  - [forEach](#foreach)
- [Functions](#functions)
  - [Function Declaration](#function-declaration)
  - [Function Expression / Anoymous Function](#function-expression--anoymous-function)
  - [Arrow Function](#arrow-function)
- [Objects](#objects)
  - [Object Method (Function)](#object-method-function)
- [Sets](#sets)
- [Maps](#maps)
- [DOM Manipulation](#dom-manipulation)

JS contains the following datatypes : `Number, String, Boolean, Undefined, Null`. JS also contains the following falsy Values : `0, '',undefined, null, NaN`.

---
Javascript is run by placing the `script` element below in the `body` element of the `HTML` file for a project.
```javascript
<script type="text/javascript" src="app.js"></script>
// src is "source" of js **file**
```

# Basic Commands
| Command                                  | Definition                                                                                 |
| :--------------------------------------- | ------------------------------------------------------------------------------------------ |
| `'use strict';`                          | turns on strict mode, which doesn't allow silent errors; insert at top of javascript file. |
| `console.log('Hello World!')`            | displays output (text, variable, etc.) to the browser's console log                        |
| `prompt("What is todays' date?")`        | displays browser prompt for user input; can be stored as a variable                        |
| `console.log(typeof variableName)`       | displays the variable's data type to the browser's console log                             |
| `alert('Hello World!')`                  | displays browser alert with output (text, variable, etc.)                                  |
| `Number('1990')`                         | converts the string into an integer.                                                       |
| `String(1991)`                           | converts the integer into a string.                                                        |
| `document.querySelector('html-element')` | accesses the HTML element passed to allow for DOM manipulation.                            |

Use `const` when you want to define an *immutable* variable, this should usually be the first choice of defining a variable. Use `let` instead of `var`! `let` will define a *mutable* variable; also `let` is "block" scoped, while `var` is "function" scoped. 

```javascript
// defines an immutable object
const birthYear = 1990;

// defines a mutable object
let superHero = 'Spider-Man';

// created a variable but remains undefined
let defeats; 

// can define variable, but this creates the variable in the "global" object and SHOULD BE AVOIDED!
realName = 'Peter Parker';
```
# Operators 
`Assignment` operators are designed to update a variable with shorthand syntax. 

```javascript
let counter = 9 + 1;
// counter = 10
counter += 10;
// counter = 20
counter *= 4;
// counter = 80
counter++;
// counter = 81
counter--;
// counter = 80
console.log(counter);
```

`Comparison` operators are designed to provide a T/F Boolean value depending on the comparison between two variables.

```javascript
let testAnswer = 85.4;
let actualAnswer = 85.4;

console.log(testAnswer > actualAnswer);
console.log(testAnswer >= actualAnswer);
console.log(testAnswer <> actualAnswer);
```
`Logical` opertators are designed to determine the logic between variables or values.

```javascript
const hasDriversLicense = true;
const hasGoodVision = false;
const goodDriveRecord = true;

console.log(hasDriversLicense && hasGoodVision);
console.log(hasDriversLicense && hasGoodVision);
console.log(hasDriversLicense);

const shouldDrive = hasDriversLicense && hasGoodVision;
const canDrive = hasDriversLicense && (hasGoodVision || goodDriveRecord);

// qualifies based on shouldDrive parameter
if(shouldDrive) {
    console.log('You are qualified to drive.');
} else {
    console.log('Someone else should drive...');
}

// qualifies based on canDrive parameter
if(canDrive) {
    console.log('You are qualified to drive.');
} else {
    console.log('Someone else should drive...');
}
```

# Template Strings / Literals
Defines a string with passed values from defined variables using backticks. Newline incorporated automatically.

```javascript
const normalName = 'Peter Parker';
const superheroName = 'Spider-Man';

console.log(`Do you belive ${normalName} could be ${superheroName}?`);
```

# Arrays
| Command                                                 | Definition                                                                  |
| ------------------------------------------------------- | --------------------------------------------------------------------------- |
| `const friendsArray = ['Iron Man', 'Spider-Man'];`      | creates an array using bracket notation.                                    |
| `const yearsArray = new Array(1991, 1984, 2008, 2024);` | creates an array using Array method.                                        |
| `let arrayLength = array.length;`                       | calculates the length of the array specified.                               |
| `let elementIdx = array.indexOf('Value');`              | returns the index of the element in the array if existing else returns -1.  |
| `let elementTF = array.includes('Value');`              | returns boolean T/F if the element is in the array.                         |
| `let newArrayLength = array.unshift('Value');`          | adds an element to the beginning of an array; captures new length of array. |
| `let newArrayLength = array.push('Value')`              | adds an element to the end of an array; captures new length of array.       |
| `let firstElement = array.shift();`                     | removes the first element from an array; captures the element removed.      |
| `let lastElement = array.pop();`                        | removes the last element from an array; captures the element removed.       |
| `let elementsAfterSlice = array.slice(index);`          | captures selected value(s) for an array after the index.                    |
| `let arrayAsString = array.join(',');`                  | joins elements in array as a string with passed separator.                  |
| `let StringAsArray = array.split(',');`                 | splits a string into an array of elements using passed separator.           |
| `const [first, second] = object.SampleArray;`           | captures the first two elements in the object's SampleArray.                |
| ` const [, , third, fourth] = object.SampleArray;`      | captures the third and fourth elements in the object's SampleArray.         |

# Dates
| Command                          | Definition                                              |
| -------------------------------- | ------------------------------------------------------- |
| `dateValue.toTimeString()`       | converts date object to "time" string.                  |
| `dateValue.toDateString()`       | converts date object to "date" string.                  |
| `dateValue.toUTCString()`        | converts date object to "UTC" string.                   |
| `dateValue.toISOString()`        | converts date object to "ISO" string.                   |
| `dateValue.toGMTString()`        | converts date object to "GMT" string.                   |
| `dateValue.toLocaleDateString()` | converts date object to "locale" sensitive date string. |

```javascript

let todaysDate = new Date();

console.log(todaysDate);
// Thu Feb 08 2024 11:09:30 GMT-0700 (Mountain Standard Time)

console.log(todaysDate.toTimeString());
// 11:07:10 GMT-0700 (Mountain Standard Time)

console.log(todaysDate.toDateString());
// Thu Feb 08 2024

console.log(todaysDate.toUTCString());
// Thu, 08 Feb 2024 18:07:10 GMT

console.log(todaysDate.toISOString());
// 2024-02-08T18:07:10.842Z

console.log(todaysDate.toGMTString());
// 08 Feb 2024 18:07:10 GMT

console.log(todaysDate.toLocaleDateString());
// 2/8/2024
```

# If/Else Statements

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

// tenerary operator example w/direct assignment to a variable
const bill = 275;

// if bill is between $50-$300 tip 15%, else tip 20%
const tip = 50 <= bill <=300 ? bill * .15 : bill * .20;
console.log(`The bill was ${bill}, the tip was ${tip}, and the total value was ${bill+tip}.`)
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
let coffeeToo = [];

for (let i = 0; i < breakfastFoods.length; i++) {
  // reading from an array
  console.log(breakfastFoods[i]);

  // filling a new array
  coffeeToo.push(`${breakfastFoods[i]} & Coffee`);
}
```

Loop syntax can also be used when looping is simple

```javascript
for (const breakfast in breakfastFoods) {
  console.log(breakfast);
}
```

## forEach

**_forEach_** iterates over each element in an array and applies the function defined

```javascript
// array of student names
var students = ['Johnny', 'Tyler', 'Bodhi', 'Pappas'];

// iterates through each element and displays
students.forEach(function (name) {
  console.log(name);
});
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
Creates an object with `key:value` pairs. 

| Command                      |                                               |
| ---------------------------- | --------------------------------------------- |
| `Object.keys(ObjectName)`    | retrieves the keys of the object.             |
| `Object.values(ObjectName)`  | retrieves the values of the object.           |
| `Object.entries(ObjectName)` | retrieves the key, value pairs of the object. |

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
    this.bmi = Math.round((this.mass / this.height ** 2) * 100) / 100;
    return `The calculated BMI is: ${this.bmi}.`;
  },

  calcAge: function () {
    // this assigns a new element (age) to 'this' current object
    this.age = 2021 - this.birthYear;
    return `The calulacted age is: ${this.age}`;
  },
};

// calls the methods in object to calculate new elements
console.log(capt_america.calcBMI());
console.log(capt_america.calcAge());
```

# Sets
Creates an object of unique values, without duplicates. Used when creating a simple list.

| Command                               |                                                     |
| ------------------------------------- | --------------------------------------------------- |
| `new Set()`                           | creates a new and empty set.                        |
| `setElement.size`                     | returns the size of the set element.                |
| `setElement.has('Spider-Man')`        | boolean T/F if element exists or not.               |
| `setElement.add('Hawkeye')`           | adds element to the set if not existing.            |
| `setElement.delete('Winter Soldier')` | deletes the element from the set if existing.       |
| `[...setElement]`                     | spread operator spreads set elements into an array. |

```javascript
const avengerSet = new Set([
  'Hulk',
  'Black Widow',
  'Ant-Man',
  'Winter Soldier',
  'Scarlett Witch',
]);
console.log(avengerSet);

// returns the size of the set
console.log(avengerSet.size);

// returns unique elements in string
console.log(new Set('Guardians of the Galaxy'));

// boolean T/F if element exists or not
console.log(avengerSet.has('Captain America'));
console.log(avengerSet.has('Ant-Man'));

// adds elements to the set if not existing
avengerSet.add('Dr. Strange');

// deletes elements from the set if existing
avengerSet.delete('Winter Soldier');

// iterates over each element in set
for (const order of avengerSet) {
  console.log(order);
}

const spideyVillians = [
  'Green Goblin',
  'Doc Ock',
  'Sandman',
  'Green Goblin',
  'Electro',
];

// captures only the unique elements, then spreads back into an array
const uniqueVilliansSet = [...new Set(spideyVillians)];
console.log(uniqueVilliansSet);
```
# Maps
Creates an object with `key:value` pairs; can contain mixed types of elements.

| Command                                |                                                      |
| -------------------------------------- | ---------------------------------------------------- |
| `new Map()`                            | creates a new and empty Map.                         |
| `mapElement.size`                      | returns the size of the map element.                 |
| `mapElement.has(2)`                    | boolean T/F if key exists or not.                    |
| `mapElement.set(1, 'Captain America')` | adds element to the map, can be chained.             |
| `mapElement.delete(1)`                 | deletes the element with the passed key if existing. |

```javascript
const galaxyGuardians = new Map();

// different ways to add elements to a map
galaxyGuardians.set('Captain', 'Starlord');
galaxyGuardians.set(2, 'Gamora');
console.log(galaxyGuardians.set('Co-Captain', 'Rocket'));

// elements can be added inline and can set multiple elements at once
galaxyGuardians
  .set('Aliens', ['Mantis', 'Gamora', 'Rocket', 'Drax the Destroyer', 'Groot'])
  .set('Humans', ['Starlord']);

// booleans can also be mapped
galaxyGuardians.set(true, 'We are Groot.').set(false, 'I am Groot.');

// displays map with new elements added
console.log(galaxyGuardians);

// returns T/F if key exists
console.log(galaxyGuardians.has('Ronan'));

// deletes an element from a map
galaxyGuardians.delete(2);

// displays the length of the map.
console.log(galaxyGuardians.size);

// checking to see if current time is during working hours or not
const approachingPerson = 'Rocket';

console.log(
  // passes two conditionals resulting in T/F response
  // this boolean is mapped to element true/false
  galaxyGuardians.get(
    galaxyGuardians.get('Aliens').includes(approachingPerson) ||
      galaxyGuardians.get('Humans').includes(approachingPerson)
  )
);

// displays the map with new changes
console.log(galaxyGuardians);
```


# DOM Manipulation

The DOM (Document Object Model) is a structured representation of HTML documents. It allows JS to access the HTML elements and styles to maniuplate them.
  * ie. change text, HTML attributes, CSS styles, etc.
  
| Command                                     |                                                                          |
| ------------------------------------------- | ------------------------------------------------------------------------ |
| `document.querySelector('html-element')`    | accesses the HTML element passed to allow for DOM manipulation.          |
| `document.querySelectorAll('html-element')` | accesses all HTML elements as `nodes`, similar to arrays.                |
| `document.querySelector('.class-element')`  | accesses the HTML class element passed to allow for DOM manipulation.    |
| `document.querySelector('#element-id')`     | accesses the HTML id class element passed to allow for DOM manipulation. |
| `document.getElementById('element-id')`     | accesses the HTML id element passed to allow for DOM manipulation.       |

Selecting an element in a `div` element to update the text.

```javascript
let messageDiv = document.querySelector('.message');
messageDiv.textContent = `Updated text for a div element with class=message.`;

// values for input elements are captured w/the .value param
let inputElement = document.querySelector('.user-input').value;
console.log(`The current user input value: ${inputElement}!`);
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

// use 'toggle' when wanting to add/remove css class depending on it's current state
topBanner.classList.toggle('hidden');
```
