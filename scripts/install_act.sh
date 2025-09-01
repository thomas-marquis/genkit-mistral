#!/bin/bash

set -e

curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
go install github.com/rhysd/actionlint/cmd/actionlint@latest
