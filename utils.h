#ifndef UTILS_H_
#define UTILS_H_

#include <cstdint>

inline int64_t RoundUpDiv(int64_t x, int64_t y) { return (x + y - 1) / y; }

inline int64_t RoundUp(int64_t x, int64_t y) { return RoundUpDiv(x, y) * y; }

#endif