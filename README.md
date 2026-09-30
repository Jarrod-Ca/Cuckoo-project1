# Cuckoo Hashing: Implementation and Empirical Study

Programming Assignment 1, Advanced Algorithms (Track A: implementation plus
empirical study).

A cuckoo hash table in C++17, parameterised on the number of hash functions
(`d`) and slots per bucket (`b`), using tabulation hashing. It is benchmarked
against `std::unordered_map` in two experiments:

- **Threshold experiment**: the load factor at which insertion first fails, for
  d=2/b=1 and d=2/b=4, across 20 seeds and several eviction budgets, plus the
  eviction-chain length of every insert.
- **Latency experiment**: per-lookup latency percentiles (p50, p99, p99.9) for
  successful and unsuccessful lookups across a sweep of load factors, plus
  probe counts and batch throughput.

## Requirements

- An x86-64 CPU. The latency timer uses the time-stamp counter (`rdtsc`).
- A C++17 compiler. Developed with MSVC (Visual Studio) on Windows. It also
  builds with GCC 13 on Linux.
- CMake 3.16 or later. Visual Studio's "C++ CMake tools for Windows" component
  provides it.
- To regenerate the figures: Python 3 with `pandas` and `matplotlib`. Anaconda
  includes both.

## Build

### Visual Studio (Windows)

1. **File → Open → Folder**, and select the repository root. Visual Studio
   detects `CMakeLists.txt` and `CMakePresets.json` and configures itself.
2. Select **x64 Release** in the configuration dropdown.
3. Select `cuckoo.exe` as the startup item and press **Ctrl+F5**.

The executable is built to `out/build/x64-release/cuckoo.exe`.

### Command line (Linux, GCC or Clang)

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/cuckoo
```

## Run

```
cuckoo            # correctness checks, then the full experiments
cuckoo --quick    # correctness checks, then a tiny smoke test of the whole pipeline
```

Every run starts with a randomised differential test against
`std::unordered_map`. It covers four table geometries, the failed-insert
rollback path, and the reserved key. If any check fails, the program stops
before running any experiment.

The full experiments refuse to run in a Debug build, because the timings would
be meaningless. Use x64 Release. `--quick` runs in either build and uses tiny
sizes. Its output is not for analysis.

A full run took [FILL IN: about N minutes] on the machine recorded in
`results/run_info.txt`.

### Output

Written to `results/` in the repository root, wherever the executable is:

| File | Contents |
|---|---|
| `threshold_failures.csv` | One row per (d, b, budget, seed): load factor at the first failed insert |
| `threshold_chains.csv` | Eviction-chain length statistics per 0.01 load bin, per run |
| `latency.csv` | Latency percentiles, batch mean and probe counts per (structure, load, trial, hit/miss) |
| `run_info.txt` | Machine, compiler, build type, timer calibration and overhead, experiment sizes |

## Figures

From the repository root:

```
python plot_results.py
```

This writes four PNGs to `results/figures/` and prints the summary numbers used
in the report.

| Figure | Shows |
|---|---|
| `fig1_threshold.png` | Failure load factor per seed at each eviction budget |
| `fig2_chain_length.png` | Mean, p99 and max evictions per insert as load rises |
| `fig3_latency_percentiles.png` | p50 / p99 / p99.9 lookup latency against load |
| `fig4_probes_vs_time.png` | Slots examined per lookup next to measured time per lookup |

## Repository layout

| File | Purpose |
|---|---|
| `cuckoo_table.hpp` | The cuckoo hash table: tabulation hashing, insert with the eviction walk and rollback, lookup, delete, rehash and growth |
| `correctness_check.hpp` | Randomised differential test against `std::unordered_map` |
| `harness.hpp` | Both experiments: data generation, timing, CSV output |
| `main.cpp` | Runs the correctness check, then the harness |
| `plot_results.py` | Figures and summary numbers from the CSVs |
| `CMakeLists.txt`, `CMakePresets.json` | Build configuration |
| `results/` | Data and figures from the run analysed in the report |

## Notes for reproducing results

- All seeds are fixed, so a rerun on the same machine generates the same keys
  and tables. Timings still vary between runs.
- Experiment sizes (slots, seeds, lookups per trial, trials) are set in
  `harness::Settings` in `harness.hpp`.
- Key value `UINT64_MAX` is reserved as the empty-slot marker and cannot be
  stored.
- Set the power plan to High performance and close other programs before a
  full run. Record both in `results/run_info.txt`.
- Timer resolution depends on the CPU. On the machine used for the included
  results, latencies are measured in steps of about 10 ns, with about 20 ns of
  timer overhead included in every value.

## AI use

AI tools were used in this project. What was generated, what was checked and
what went wrong are described in section (d) of the report.
