#ifndef CIRCULAR_BUFFER_HPP
#define CIRCULAR_BUFFER_HPP

#include <iostream>
#include <vector>
#include <unordered_map>

namespace pneatm {

/**
 * @brief A template class representing a constant size circular buffer.
 *
 * A ciruclar buffer is a storing object that loop on his end to the first element.
 * This enable it to avoid reallocating memory as it constantly overwrite previous values.
 * 
 * @tparam T The stored types.
 */
template <typename T>
class CircularBuffer {
public:
    /**
     * @brief Construct a new CircularBuffer object
     * @param capacity The constant capacity of the buffer. (default is 0)
     */
    CircularBuffer (const unsigned int capacity = 0);

    /**
     * @brief Insert an element in the buffer.
     * @param elem The element to be inserted.
     */
    void insert (const T& elem);

    /**
     * @brief Operator [] to retrieve an element at a given index.
     * @param n The index.
     * @return T The element at index n.
     */
    T operator[] (unsigned int n) const;

    /**
     * @brief Get a pointer to the element at a given index.
     * @param n The index.
     * @return T* A pointer to the element at index n.
     */
    T* access_ptr (unsigned int n);

private:
    unsigned int capacity;
    std::vector<T> buffer;
    unsigned int currentIndex;
    
};

template <typename T>
CircularBuffer<T>::CircularBuffer (unsigned int capacity) :
    capacity (capacity),
    buffer (capacity),
    currentIndex (0)
{}

template <typename T>
void CircularBuffer<T>::insert (const T& elem) {
    buffer [currentIndex] = elem;
    currentIndex = (currentIndex + 1) % capacity;
}

template <typename T>
T CircularBuffer<T>::operator[] (unsigned int n) const {
    return buffer [(currentIndex - 1 - n + capacity) % capacity];
}

template <typename T>
T* CircularBuffer<T>::access_ptr (unsigned int n) {
    return &buffer [(currentIndex - 1 - n + capacity) % capacity];
}

}

#endif  // CIRCULAR_BUFFER_HPP