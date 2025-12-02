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

add_executable(relief_app
    src/flatten_surface.cpp
    src/main.cpp
)
target_compile_features(relief_app PRIVATE cxx_std_23)
target_link_libraries(relief_app PRIVATE
    fmt::fmt
    CGAL::CGAL
    CLI11::CLI11
    Eigen3::Eigen
    geometry-central
    igl::core
    igl_restricted::triangle
)
target_include_directories(relief_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# if(MSVC)
#     target_compile_options(relief_app PRIVATE /W4 /permissive- /WX)
# else()
#     target_compile_options(relief_app PRIVATE -Wall -Wextra -Wpedantic -Werror)
# endif()
