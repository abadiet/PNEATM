cmake -G "MinGW Makefiles" -B "./build" -DX64_BITS=OFF .
cd build
make
cp ./liblrneat.a ../lib/liblrneat.a
cd ..
