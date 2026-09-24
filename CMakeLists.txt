cmake_minimum_required(VERSION 3.24)

project(relief VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CPM.cmake)

find_package(Eigen3 REQUIRED)
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE ON)
find_package(CGAL REQUIRED)

CPMAddPackage(
    NAME fmt
    GIT_TAG 12.1.0
    GITHUB_REPOSITORY fmtlib/fmt
    OPTIONS
        "FMT_DOC OFF"
        "FMT_TEST OFF"
        "FMT_INSTALL OFF"
)
CPMAddPackage("gh:CLIUtils/CLI11#v2.6.1")
CPMAddPackage(
    NAME geometry-central
    GIT_TAG master
    GITHUB_REPOSITORY liukaizheng/geometry-central
    OPTIONS
        "SUITESPARSE OFF"
)
CPMAddPackage(
    NAME libigl
    GIT_TAG main
    GITHUB_REPOSITORY liukaizheng/libigl
    OPTIONS
        "LIBIGL_RESTRICTED_TRIANGLE ON"
)
# CPMAddPackage("gh:liukaizheng/gpf-algorithm#main")
CPMAddPackage(
    NAME gpf-algorithm
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../gpf-algorithm
    OPTIONS "BUILD_TESTING OFF" "GPF_ALGORITHM_BUILD_TESTS OFF"
)

add_library(relief_surface
    src/flatten_surface.cpp
    src/fit_on_surface.cpp
)
target_compile_features(relief_surface PUBLIC cxx_std_23)
target_include_directories(relief_surface PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(relief_surface PUBLIC Eigen3::Eigen gpf::algorithm
    PRIVATE fmt::fmt igl::core)

add_executable(relief_app src/main.cpp)
target_compile_features(relief_app PRIVATE cxx_std_23)
target_link_libraries(relief_app PRIVATE
    relief_surface
    fmt::fmt
    CGAL::CGAL
    CLI11::CLI11
    Eigen3::Eigen
    geometry-central
    igl::core
    igl_restricted::triangle
    gpf::algorithm
)
target_include_directories(relief_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

option(RELIEF_ENABLE_SANITIZERS "Instrument relief targets with AddressSanitizer and UBSan" OFF)
if(RELIEF_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(relief_surface PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(relief_surface PUBLIC -fsanitize=address,undefined)
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        CPMAddPackage(
            NAME googletest
            GITHUB_REPOSITORY google/googletest
            GIT_TAG v1.17.0
            OPTIONS "INSTALL_GTEST OFF" "BUILD_GMOCK OFF"
        )
    endif()
    foreach(module surface_patch fit_on_surface image_relief)
        # White-box fixtures include fit_on_surface.cpp directly so its private
        # stages stay in one file. Do not also link a second copy from the library.
        add_executable(test_${module}
            tests/surface/test_${module}.cpp
            src/flatten_surface.cpp
        )
        target_link_libraries(test_${module} PRIVATE Eigen3::Eigen gpf::algorithm fmt::fmt igl::core GTest::gtest_main)
        target_include_directories(test_${module} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        if(RELIEF_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            target_compile_options(test_${module} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(test_${module} PRIVATE -fsanitize=address,undefined)
        endif()
        set(test_working_directory ${CMAKE_CURRENT_BINARY_DIR}/test-work/${module})
        file(MAKE_DIRECTORY ${test_working_directory})
        add_test(NAME test_${module} COMMAND test_${module})
        set_tests_properties(test_${module} PROPERTIES WORKING_DIRECTORY ${test_working_directory})
    endforeach()
endif()

# if(MSVC)
#     target_compile_options(relief_app PRIVATE /W4 /permissive- /WX)
# else()
#     target_compile_options(relief_app PRIVATE -Wall -Wextra -Wpedantic -Werror)
# endif()
