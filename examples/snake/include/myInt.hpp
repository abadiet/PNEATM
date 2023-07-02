#ifndef MYINT_HPP
#define MYINT_HPP

#include <iostream>

class myInt {
private:
    int value;

public:
    myInt(int value = 0) : value(value) {}

    myInt operator*(const float scalar) const {
        return myInt ((int) ((float) value * scalar));
    }

    void operator+=(const myInt& other) {
        value += other.value;
    }

    friend std::ostream& operator<<(std::ostream& os, const myInt& x) {
        os << x.value;
        return os;
    }

    operator int() const {
        return value;
    }
};

#endif  // MYINT_HPP