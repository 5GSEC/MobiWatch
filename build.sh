#!/usr/bin/env bash

docker build --tag localhost:5000/deep-watch-rapp:latest .
docker push localhost:5000/deep-watch-rapp

