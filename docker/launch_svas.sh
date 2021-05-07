#!/bin/sh

# Makeing sure that you are in the home directory of SVAS.
docker run -i -v $PWD:/mnt/svas tigerchange/svas /bin/sh