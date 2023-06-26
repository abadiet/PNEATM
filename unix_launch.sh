#!/bin/bash

cmake -B ./build/ -S ./
cd ./build/
make
cp ./libpneatm.a ../lib/PNEATM/libpneatm.a