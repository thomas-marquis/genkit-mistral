#!/bin/bash

set -e

WORKFLOW="${1:-ci}"

./bin/act pull_request \
  -W ".github/workflows/${WORKFLOW}.yml" -j quality \
  -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:act-22.04 \
  --artifact-server-path ./artifacts
