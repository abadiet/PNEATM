#include "myTypes.hpp"

myInt myInt::operator*(const double scalar) const {
    return myInt(static_cast<int>(static_cast<double>(value) * scalar));
}

void myInt::operator+=(const myInt& other) {
    value += other.value;
}

std::ostream& operator<<(std::ostream& os, const myInt& x) {
    os << x.value;
    return os;
}

myInt::operator int() const {
    return value;
}

myInt::operator myFloat() const {
    std::cout << "ERROR: This function should not be used!" << std::endl;
    return myFloat();
}

myFloat myFloat::operator*(const double scalar) const {
    return myFloat(static_cast<float>(static_cast<double>(value) * scalar));
}

void myFloat::operator+=(const myFloat& other) {
    value += other.value;
}

std::ostream& operator<<(std::ostream& os, const myFloat& x) {
    os << x.value;
    return os;
}

myFloat::operator float() const {
    return value;
}

myFloat::operator myInt() const {
    std::cout << "ERROR: This function should not be used!" << std::endl;
    return myInt();
}
