cmake -G "MinGW Makefiles" -B "./build" -DX64_BITS=OFF .
cd build
make
cp ./libpneatm.a ../lib/libpneatm.a
cd ..