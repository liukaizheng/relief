# Repository Guidelines

## Project Structure & Module Organization
Keep production sources under `src/` and public headers in `include/relief/`. Group features by module (`geometry/`, `io/`, etc.) and expose headers through a matching directory tree to keep include paths predictable. Executables or quick experiments belong in `apps/` with thin `main` files that delegate to reusable library code. Integration tests and fixtures live under `tests/`, mirroring the source tree so `tests/geometry/PlaneTests.cpp` clearly exercises `src/geometry/Plane.cpp`. Shared assets such as sample meshes or configuration templates should be stored in `assets/` and referenced relative to the repo root for portability.

## Build, Test, and Development Commands
Configure once per build type: `cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug`. Use Release builds for benchmarks by swapping the build directory. Compile with `cmake --build build/debug -j$(sysctl -n hw.ncpu)`; this automatically chooses Ninja or Make based on your generator. Run the full test suite via `ctest --test-dir build/debug --output-on-failure`. During development, `cmake --build build/debug --target format` keeps formatting consistent if the target is defined in `CMakeLists.txt`.

## Coding Style & Naming Conventions
Stick to C++20 features that are supported by Clang and GCC on macOS/Linux. Format all sources with `clang-format` using the repository `.clang-format` (Google style base with project-specific tweaks). Prefer `snake_case` for files, `PascalCase` for classes/types, and `camelCase` for functions and variables. Keep headers self-contained, include what you use, and order includes as: project, third-party, standard. Document non-obvious behavior with Doxygen-style comments so generated docs stay accurate.

## Testing Guidelines
Use GoogleTest for unit coverage; new modules require at least one focused test fixture named `<Module>Test`. Name test binaries `test_<module>` and register them with `add_test` so `ctest` discovers them automatically. Favor fast deterministic unit tests; mark longer scenarios with the `Slow` label and gate them behind `ctest -L Slow`. Maintain >85% line coverage for core logic and provide regression tests for every bug fix.

## Commit & Pull Request Guidelines
Write commits in the imperative mood (`Fix mesh normal handling`). Keep related changes together and avoid mixing refactors with functional edits. Pull requests should describe motivation, implementation notes, and any trade-offs; link to tracking issues and include `ctest` output or screenshots for UI-facing pieces. Ensure CI passes before requesting review, assign at least one maintainer, and respond to feedback within one business day to keep momentum.
