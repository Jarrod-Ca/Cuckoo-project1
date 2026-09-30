// main.cpp
// Runs the correctness checks and stops if any fail.

#include "correctness_check.hpp"
#include "cuckoo_table.hpp"

#include <iostream>

int main() {
    if (!cuckoo::test::run_correctness_checks(std::cerr)) {
        std::cerr << "Correctness check FAILED - not running experiments.\n";
        return 1;
    }
    std::cerr << "All correctness checks passed.\n";

    return 0;
}
