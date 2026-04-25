// Project
#include <Util/util.h>

void util::to_lowercase(std::string& str) {
    for (char& c : str) {
        c = std::tolower(c);
    }
}

bool util::starts_with(const std::string& str, const std::string& prefix, bool case_insensitive) {
    std::string tmp_str = str;
    std::string tmp_prefix = prefix;
    if (case_insensitive) {
        to_lowercase(tmp_str);
        to_lowercase(tmp_prefix);
    }
    return tmp_str.find(tmp_prefix) == 0;
}

bool util::ends_with(const std::string& str, const std::string& suffix, bool case_insensitive) {
    if (str.size() < suffix.size()) return false;
    std::string tail = str.substr(str.size() - suffix.size());
    std::string suf = suffix;
    if (case_insensitive) {
        to_lowercase(tail);
        to_lowercase(suf);
    }
    return tail == suf;
}