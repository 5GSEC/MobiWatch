#!/usr/bin/env bash

sudo make image/deep-watch-rapp
sudo docker push localhost:5000/deep-watch-rapp
