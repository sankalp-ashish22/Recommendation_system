const cp = require('child_process');

// console.log(JSON.parse(cp.execSync('python3 hello.py john').toString().trim().replaceAll("'",'"')));
const { execSync } = require("child_process");

let s;
eval(`s = ${execSync("python ./hello2.py Avatar").toString()}`);
var names = s[0];
console.log(s);







