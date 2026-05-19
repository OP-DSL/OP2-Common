#include "SafeLong.h"

// ----------------------
// Arithmetic Operators
// ----------------------

// (1) SafeLong op SafeLong
SafeLong operator+(const SafeLong &lhs, const SafeLong &rhs) {
    return SafeLong(lhs.get() + rhs.get());
}

SafeLong operator-(const SafeLong &lhs, const SafeLong &rhs) {
    return SafeLong(lhs.get() - rhs.get());
}

SafeLong operator*(const SafeLong &lhs, const SafeLong &rhs) {
    return SafeLong(lhs.get() * rhs.get());
}

SafeLong operator/(const SafeLong &lhs, const SafeLong &rhs) {
    if (rhs.get() == 0) {
        throw std::domain_error("Division by zero");
    }
    return SafeLong(lhs.get() / rhs.get());
}

SafeLong operator%(const SafeLong &lhs, const SafeLong &rhs) {
    if (rhs.get() == 0) {
        throw std::domain_error("Modulo by zero");
    }
    return SafeLong(lhs.get() % rhs.get());
}

// (2) SafeLong op int
SafeLong operator+(const SafeLong &lhs, int rhs) {
    long long res = lhs.get() + rhs;
    return SafeLong(res);
}

SafeLong operator-(const SafeLong &lhs, int rhs) {
    long long res = lhs.get() - rhs;
    return SafeLong(res);
}

SafeLong operator*(const SafeLong &lhs, int rhs) {
    long long res = lhs.get() * rhs;
    return SafeLong(res);
}

SafeLong operator/(const SafeLong &lhs, int rhs) {
    if (rhs == 0) {
        throw std::domain_error("Division by zero");
    }
    long long res = lhs.get() / rhs;
    return SafeLong(res);
}

SafeLong operator%(const SafeLong &lhs, int rhs) {
    if (rhs == 0) {
        throw std::domain_error("Modulo by zero");
    }
    long long res = lhs.get() % rhs;
    return SafeLong(res);
}

// (3) int op SafeLong
SafeLong operator+(int lhs, const SafeLong &rhs) {
    return rhs + lhs;
}

SafeLong operator-(int lhs, const SafeLong &rhs) {
    long long res = lhs - rhs.get();
    return SafeLong(res);
}

SafeLong operator*(int lhs, const SafeLong &rhs) {
    return rhs * lhs;
}

SafeLong operator/(int lhs, const SafeLong &rhs) {
    if (rhs.get() == 0) {
        throw std::domain_error("Division by zero");
    }
    long long res = lhs / rhs.get();
    return SafeLong(res);
}

SafeLong operator%(int lhs, const SafeLong &rhs) {
    if (rhs.get() == 0) {
        throw std::domain_error("Modulo by zero");
    }
    long long res = lhs % rhs.get();
    return SafeLong(res);
}

// Unary minus operator
SafeLong operator-(const SafeLong &s) {
    return SafeLong(-s.get());
}

// size_t operators
SafeLong operator*(std::size_t lhs, const SafeLong &rhs) {
    if (rhs.get() < 0) {
        throw std::overflow_error("Cannot multiply size_t with negative SafeLong");
    }
    
    unsigned long long ull_lhs = lhs;
    unsigned long long ull_rhs = static_cast<unsigned long long>(rhs.get());
    
    if (ull_rhs != 0 && (ull_lhs * ull_rhs) / ull_rhs != ull_lhs) {
        throw std::overflow_error("Multiplication would overflow");
    }
    
    return SafeLong(static_cast<long long>(ull_lhs * ull_rhs));
}

SafeLong operator*(const SafeLong &lhs, std::size_t rhs) {
    return rhs * lhs;
}

SafeLong operator+(std::size_t lhs, const SafeLong &rhs) {
    if (rhs.get() < 0) {
        throw std::overflow_error("Cannot add size_t with negative SafeLong");
    }
    
    unsigned long long ull_lhs = lhs;
    unsigned long long ull_rhs = static_cast<unsigned long long>(rhs.get());
    
    if (ull_lhs + ull_rhs > static_cast<unsigned long long>(std::numeric_limits<long long>::max())) {
        throw std::overflow_error("Addition would overflow");
    }
    
    return SafeLong(static_cast<long long>(ull_lhs + ull_rhs));
}

SafeLong operator+(const SafeLong &lhs, std::size_t rhs) {
    return rhs + lhs;
}

// ----------------------
// Logical (Comparison) Operators
// ----------------------

bool operator==(const SafeLong &lhs, const SafeLong &rhs) {
    return lhs.get() == rhs.get();
}

bool operator!=(const SafeLong &lhs, const SafeLong &rhs) {
    return lhs.get() != rhs.get();
}

bool operator<(const SafeLong &lhs, const SafeLong &rhs) {
    return lhs.get() < rhs.get();
}

bool operator<=(const SafeLong &lhs, const SafeLong &rhs) {
    return lhs.get() <= rhs.get();
}

bool operator>(const SafeLong &lhs, const SafeLong &rhs) {
    return lhs.get() > rhs.get();
}

bool operator>=(const SafeLong &lhs, const SafeLong &rhs) {
    return lhs.get() >= rhs.get();
}

// Comparisons between SafeLong and int
bool operator==(const SafeLong &lhs, int rhs) {
    return lhs.get() == rhs;
}

bool operator==(int lhs, const SafeLong &rhs) {
    return lhs == rhs.get();
}

bool operator!=(const SafeLong &lhs, int rhs) {
    return lhs.get() != rhs;
}

bool operator!=(int lhs, const SafeLong &rhs) {
    return lhs != rhs.get();
}

bool operator<(const SafeLong &lhs, int rhs) {
    return lhs.get() < rhs;
}

bool operator<(int lhs, const SafeLong &rhs) {
    return lhs < rhs.get();
}

bool operator<=(const SafeLong &lhs, int rhs) {
    return lhs.get() <= rhs;
}

bool operator<=(int lhs, const SafeLong &rhs) {
    return lhs <= rhs.get();
}

bool operator>(const SafeLong &lhs, int rhs) {
    return lhs.get() > rhs;
}

bool operator>(int lhs, const SafeLong &rhs) {
    return lhs > rhs.get();
}

bool operator>=(const SafeLong &lhs, int rhs) {
    return lhs.get() >= rhs;
}

bool operator>=(int lhs, const SafeLong &rhs) {
    return lhs >= rhs.get();
}

// Stream operator
std::ostream& operator<<(std::ostream &os, const SafeLong &s) {
    os << s.get();
    return os;
} 