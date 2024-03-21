#!/bin/bash
sudo kubectl logs $(sudo kubectl get pods -o name -n ricxapp | grep "deepwatch-xapp") -n ricxapp -f
