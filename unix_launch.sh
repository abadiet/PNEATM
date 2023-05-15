#!/bin/bash

cmake -B ./build/ -S ./
cd ./build/
make
cp ./liblrneat.a ../lib/liblrneat.a