#!/usr/bin/env bash

# Build image
docker build --tag=ids721finalproject .

# List docker images
docker image ls

# Run flask app
docker run -p 8000:80 ids721finalproject
