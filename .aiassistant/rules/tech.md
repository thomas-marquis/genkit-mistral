---
apply: always
---

## General rules

This repo is split into 2 parts:
* `mistralclient` package. In the future, this package will be moved in its own go library (but not now)
* `mistral` package (which depends on the previous one)
* Write all (code, comments, doc...) in English
* Write the less comment possible, except for public functions/structs/methods documentation

`internal` contains shared non-public code

## Unit tests

* Each test must be structured with the `// Given`, `// When`, `// Then` pattern.
* Use the package `github.com/stretchr/testify/assert` for assertions
* tests' package name must be formated like this: "<thepackage>_test"
* Test only the public functions/methods
* Test function names must be formatted as: "Test_<functionUnderTest>_Should<return/doSomething>_When<thisHappens>"