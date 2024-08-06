#ifndef UTIL_H
#define UTIL_H

// std
#include <string>

namespace util {

    void to_lowercase(std::string& str);
    bool starts_with(const std::string& str, const std::string& prefix, bool case_insensitive = true);
    bool ends_with(const std::string& str, const std::string& suffix, bool case_insensitive = true);

}

#endif // UTIL_H
