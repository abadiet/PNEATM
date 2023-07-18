#ifndef MYTYPES_HPP
#define MYTYPES_HPP

#include <iostream>

class myFloat;  // Forward declaration

class myInt {
private:
    int value;

public:
    myInt(int value = 0) : value(value) {}

    myInt operator*(const double scalar) const;

    void operator+=(const myInt& other);

    friend std::ostream& operator<<(std::ostream& os, const myInt& x);

    operator int() const;

    operator myFloat() const;
};

class myFloat {
private:
    float value;

public:
    myFloat(float value = 0.0f) : value(value) {}

    myFloat operator*(const double scalar) const;

    void operator+=(const myFloat& other);

    friend std::ostream& operator<<(std::ostream& os, const myFloat& x);

    operator float() const;

    operator myInt() const;
};

#endif  // MYTYPES_HPP
