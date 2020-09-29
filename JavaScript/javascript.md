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

|              Command              |                                                                     |
| :-------------------------------: | ------------------------------------------------------------------- |
|   `console.log('Hello World!')`   | displays output (text, variable, etc.) to the browser's console log |
|      `alert('Hello World!')`      | displays browser alert with output (text, variable, etc.)           |
| `prompt('What is todays' date?')` | displays browser prompt for user input; can be stored as a variable |
|  `console.log(typeof variable)`   | displays the variable's data type to the browser's console log      |
|                ``                 |                                                                     |
|                ``                 |                                                                     |

defines a variable
```javascript
var variable_name = "Chris";
```

if/else statement displaying message to the console log depending on boolean isMale variable
```javascript
// creating boolean value isMale
var isMale = true;

if (isMale) {
    console.log(isMale + ' , he is male.')
} else {
    console.log(isMale + ' , she is female.')
}
```

ternary operators more efficiently define an if/else statement
```javascript
var alias = 'Peter Parker';

// ternary operators are setups as: conditional ? programming if TRUE : programming if FALSE;
alias === 'Peter Parker' ? console.log('I think that dude is Spiderman?') : console.log('Their name is ' + alias + '.');

//  ternary operators can also be assigned to variables
var criminalsJailed = 100;

// conditional ? programming if TRUE assigned : programming if FALSE assigned; 
var ability = criminalsJailed >= 80 ? 'novice' : 'veteran';

console.log('That superhero is a well known ' + ability);
```

switch statements provide if/else comparisons over multiple cases
```javascript
var job = 'photographer';
var criminalsJailed = 100;

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
switch(true) {
    case criminalsJailed <= 100:
        console.log('Friendly Neighborhood Spiderman');

    case criminalsJailed > 100 && criminalsJailed <= 500:
        console.log('The Amazing Spiderman');

    case criminalsJailed > 500:
        console.log('The Superior Spiderman');

    default:
        console.log("The Spider Menace");
}
```