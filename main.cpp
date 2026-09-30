// main.cpp
// Runs the correctness checks and stops if any fail.

#include "correctness_check.hpp"
#include "cuckoo_table.hpp"
#include "harness.hpp"
#include <iostream>
#include <cstring>
int main(int argc, char** argv) {
    if (!cuckoo::test::run_correctness_checks(std::cerr)) {
        std::cerr << "Correctness check FAILED - not running experiments.\n";
        return 1;
    }
    std::cerr << "All correctness checks passed.\n";

    // --quick: tiny sizes, checks the whole pipeline in seconds. Not for the report.
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    const harness::Settings settings = quick ? harness::quick_settings() : harness::Settings{};
    return harness::run_all(settings, quick, std::cerr);
}
