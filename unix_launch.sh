#!/bin/bash

cmake -B ./build/ -S ./
cd ./build/
make
cp ./libvrneat.a ../lib/libvrneat.a