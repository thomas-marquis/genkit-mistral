#!/bin/bash

set -e

golangci-lint run  --timeout=5m ./...
