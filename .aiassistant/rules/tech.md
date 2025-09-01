---
apply: always
---

## General rules

This repo is split into 2 parts:
* `mistralclient` package. In the future, this package will be moved in its own go library (but not now)
* `mistral` package (which depends on the previous one)

`internal` contains shared non-public code

## Unit tests

Each test must be structured with the `// Given`, `// When`, `// Then` pattern.
