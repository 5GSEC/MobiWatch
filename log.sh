#!/bin/bash
sudo kubectl logs $(sudo kubectl get pods -o name -n ricxapp | grep "mobiwatch-xapp") -n ricxapp -f
