#pragma once
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cstddef>

class SafeLong {
public:
    // Default constructor
    SafeLong() : value(0LL) {}

    // Constructors
    SafeLong(long long v) : value(v) {}
    SafeLong(int v) : value(v) {}
    SafeLong(std::size_t v) : value(v) {}
    SafeLong(std::ptrdiff_t v) : value(static_cast<long long>(v)) {}

    // Getter (for internal use)
    long long get() const { return value; }

    // Implicit conversion to int with range check
    operator int() const {
        if (value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max()) {
            throw std::overflow_error("Conversion to int would lose data");
        }
        return static_cast<int>(value);
    }

    // Implicit conversion to longint with range check
    operator long int() const {
        if (value < std::numeric_limits<long int>::min() ||
            value > std::numeric_limits<long int>::max()) {
            throw std::overflow_error("Conversion to long int would lose data");
        }
        return static_cast<long int>(value);
    }

    // Implicit conversion to size_t with range check
    operator std::size_t() const {
        if (value < 0) {
            throw std::overflow_error("Cannot convert negative value to size_t");
        }
        return static_cast<std::size_t>(value);
    }
    
    operator long long unsigned int() const {
        if (value < 0) {
            throw std::overflow_error("Cannot convert negative value to long long unsigned int");
        }
        return static_cast<long long unsigned int>(value);
    }

    // Implicit conversion to long long int
    operator long long() const { return value; }

    // Prefix increment operator (++x)
    SafeLong& operator++() {
        value++;
        return *this;
    }

    // Postfix increment operator (x++)
    SafeLong operator++(int) {
        SafeLong temp = *this;
        value++;
        return temp;
    }

    // Compound assignment operators
    SafeLong& operator+=(const SafeLong &other) {
        value += other.value;
        return *this;
    }
    SafeLong& operator-=(const SafeLong &other) {
        value -= other.value;
        return *this;
    }
    SafeLong& operator*=(const SafeLong &other) {
        value *= other.value;
        return *this;
    }
    SafeLong& operator/=(const SafeLong &other) {
        if (other.value == 0) {
            throw std::domain_error("Division by zero");
        }
        value /= other.value;
        return *this;
    }

    SafeLong& operator/=(const int &other) {
        if (other == 0) {
            throw std::domain_error("Division by zero");
        }
        value /= other;
        return *this;
    }
    SafeLong& operator%=(const SafeLong &other) {
        if (other.value == 0) {
            throw std::domain_error("Modulo by zero");
        }
        value %= other.value;
        return *this;
    }

    // Compound assignment operator for int
    SafeLong& operator+=(int other) {
        if (other > 0 && value > std::numeric_limits<long long>::max() - static_cast<long long>(other)) {
            throw std::overflow_error("Addition would overflow");
        } else if (other < 0 && value < std::numeric_limits<long long>::min() - static_cast<long long>(other)) {
            throw std::overflow_error("Addition would overflow");
        }
        value += static_cast<long long>(other);
        return *this;
    }

    // Compound assignment operator for size_t
    SafeLong& operator+=(std::size_t other) {
        if (value > std::numeric_limits<long long>::max() - static_cast<long long>(other)) {
            throw std::overflow_error("Addition would overflow");
        }
        value += static_cast<long long>(other);
        return *this;
    }

    // Compound assignment operator for long int
    SafeLong& operator/=(long int other) {
        if (other == 0) {
            throw std::domain_error("Division by zero");
        }
        value /= other;
        return *this;
    }

private:
    long long value;
};

// Arithmetic Operators declarations
SafeLong operator+(const SafeLong &lhs, const SafeLong &rhs);
SafeLong operator-(const SafeLong &lhs, const SafeLong &rhs);
SafeLong operator*(const SafeLong &lhs, const SafeLong &rhs);
SafeLong operator/(const SafeLong &lhs, const SafeLong &rhs);
SafeLong operator%(const SafeLong &lhs, const SafeLong &rhs);

SafeLong operator+(const SafeLong &lhs, int rhs);
SafeLong operator-(const SafeLong &lhs, int rhs);
SafeLong operator*(const SafeLong &lhs, int rhs);
SafeLong operator/(const SafeLong &lhs, int rhs);
SafeLong operator%(const SafeLong &lhs, int rhs);

SafeLong operator+(int lhs, const SafeLong &rhs);
SafeLong operator-(int lhs, const SafeLong &rhs);
SafeLong operator*(int lhs, const SafeLong &rhs);
SafeLong operator/(int lhs, const SafeLong &rhs);
SafeLong operator%(int lhs, const SafeLong &rhs);

SafeLong operator-(const SafeLong &s);

SafeLong operator*(std::size_t lhs, const SafeLong &rhs);
SafeLong operator*(const SafeLong &lhs, std::size_t rhs);
SafeLong operator+(std::size_t lhs, const SafeLong &rhs);
SafeLong operator+(const SafeLong &lhs, std::size_t rhs);

// Comparison operators declarations
bool operator==(const SafeLong &lhs, const SafeLong &rhs);
bool operator!=(const SafeLong &lhs, const SafeLong &rhs);
bool operator<(const SafeLong &lhs, const SafeLong &rhs);
bool operator<=(const SafeLong &lhs, const SafeLong &rhs);
bool operator>(const SafeLong &lhs, const SafeLong &rhs);
bool operator>=(const SafeLong &lhs, const SafeLong &rhs);

bool operator==(const SafeLong &lhs, int rhs);
bool operator==(int lhs, const SafeLong &rhs);
bool operator!=(const SafeLong &lhs, int rhs);
bool operator!=(int lhs, const SafeLong &rhs);
bool operator<(const SafeLong &lhs, int rhs);
bool operator<(int lhs, const SafeLong &rhs);
bool operator<=(const SafeLong &lhs, int rhs);
bool operator<=(int lhs, const SafeLong &rhs);
bool operator>(const SafeLong &lhs, int rhs);
bool operator>(int lhs, const SafeLong &rhs);
bool operator>=(const SafeLong &lhs, int rhs);
bool operator>=(int lhs, const SafeLong &rhs);

// size_t compound assignment operator (non-member)
inline std::size_t& operator+=(std::size_t& lhs, const SafeLong& rhs) {
    std::size_t rhs_size = static_cast<std::size_t>(rhs.get());
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs_size) {
        throw std::overflow_error("Addition would overflow size_t");
    }
    lhs = lhs + rhs_size;
    return lhs;
}

// long int compound division operator (non-member)
inline long int& operator/=(long int& lhs, const SafeLong& rhs) {
    if (rhs.get() == 0) {
        throw std::domain_error("Division by zero");
    }
    lhs = lhs / static_cast<long int>(rhs);
    return lhs;
}

// Stream operator declaration
std::ostream& operator<<(std::ostream &os, const SafeLong &s);