# Hands-On CMake and Make Tutorial

## Table of Contents
1. [Setting Up Your Environment](#setting-up-your-environment)
2. [Simple CMake Project](#simple-cmake-project)
3. [Intermediate CMake Project](#intermediate-cmake-project)
4. [Advanced CMake Project](#advanced-cmake-project)
5. [Using Make Directly](#using-make-directly)
6. [Integration Examples](#integration-examples)
7. [Troubleshooting Exercises](#troubleshooting-exercises)

## Setting Up Your Environment

Before starting, ensure you have the necessary tools installed:

```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake

# On CentOS/RHEL/Fedora
sudo dnf install gcc gcc-c++ make cmake

# On macOS with Homebrew
brew install cmake

# Verify installations
cmake --version
gcc --version
make --version
```

## Simple CMake Project

Let's start with a minimal CMake project to understand the basics.

### Exercise 1: Hello World with CMake

Create the following directory structure:
```
hello_cmake/
├── CMakeLists.txt
└── src/
    └── main.cpp
```

**File: hello_cmake/src/main.cpp**
```cpp
#include <iostream>

int main() {
    std::cout << "Hello, CMake World!" << std::endl;
    return 0;
}
```

**File: hello_cmake/CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.10)
project(HelloCMake VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hello src/main.cpp)
```

**Build Instructions:**
```bash
cd hello_cmake
mkdir build
cd build
cmake ..
make
./hello
```

Expected output: `Hello, CMake World!`

### Exercise 2: Adding a Library

Extend the previous example to include a custom library.

Directory structure:
```
hello_cmake/
├── CMakeLists.txt
├── include/
│   └── hello.hpp
├── src/
│   ├── main.cpp
│   └── hello.cpp
```

**File: hello_cmake/include/hello.hpp**
```cpp
#ifndef HELLO_HPP
#define HELLO_HPP

void say_hello();

#endif
```

**File: hello_cmake/src/hello.cpp**
```cpp
#include <iostream>
#include "hello.hpp"

void say_hello() {
    std::cout << "Hello from a library!" << std::endl;
}
```

**File: hello_cmake/src/main.cpp**
```cpp
#include "hello.hpp"

int main() {
    say_hello();
    return 0;
}
```

**Updated File: hello_cmake/CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.10)
project(HelloCMake VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create a library
add_library(hellolib src/hello.cpp)

# Specify include directories for the library
target_include_directories(hellolib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create executable
add_executable(hello src/main.cpp)

# Link the library to the executable
target_link_libraries(hello PRIVATE hellolib)
```

**Build Instructions:**
```bash
cd hello_cmake
mkdir -p build
cd build
cmake ..
make
./hello
```

Expected output: `Hello from a library!`

## Intermediate CMake Project

### Exercise 3: Multi-Directory Project with Tests

Create a more complex project structure:
```
calculator/
├── CMakeLists.txt
├── include/
│   └── calculator/
│       ├── math.hpp
│       └── version.hpp
├── src/
│   ├── CMakeLists.txt
│   ├── calculator.cpp
│   └── main.cpp
├── tests/
│   ├── CMakeLists.txt
│   └── test_calculator.cpp
└── cmake/
    └── version.h.in
```

**File: calculator/cmake/version.h.in**
```cpp
#pragma once

#define CALCULATOR_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define CALCULATOR_VERSION_MINOR @PROJECT_VERSION_MINOR@
#define CALCULATOR_VERSION_PATCH @PROJECT_VERSION_PATCH@
#define CALCULATOR_VERSION "@PROJECT_VERSION@"
```

**File: calculator/CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.15)
project(Calculator VERSION 1.0.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate version header
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/version.h.in"
    "${PROJECT_BINARY_DIR}/generated/version.h"
    @ONLY
)

# Add subdirectories
add_subdirectory(src)

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

**File: calculator/src/CMakeLists.txt**
```cmake
# Create static library
add_library(calclib
    calculator.cpp
)

target_include_directories(calclib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_BINARY_DIR}/generated
)

# Create executable
add_executable(calc main.cpp)
target_link_libraries(calc PRIVATE calclib)

# Installation
install(TARGETS calc calclib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
```

**File: calculator/include/calculator/math.hpp**
```cpp
#ifndef CALCULATOR_MATH_HPP
#define CALCULATOR_MATH_HPP

namespace calc {
    double add(double a, double b);
    double subtract(double a, double b);
    double multiply(double a, double b);
    double divide(double a, double b);
}

#endif
```

**File: calculator/src/calculator.cpp**
```cpp
#include "calculator/math.hpp"
#include <stdexcept>

namespace calc {
    double add(double a, double b) {
        return a + b;
    }

    double subtract(double a, double b) {
        return a - b;
    }

    double multiply(double a, double b) {
        return a * b;
    }

    double divide(double a, double b) {
        if (b == 0) {
            throw std::domain_error("Division by zero");
        }
        return a / b;
    }
}
```

**File: calculator/src/main.cpp**
```cpp
#include "calculator/math.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: calc <num1> <operator> <num2>\n";
        return 1;
    }

    double a = std::stod(argv[1]);
    std::string op = argv[2];
    double b = std::stod(argv[3]);
    double result;

    if (op == "+") {
        result = calc::add(a, b);
    } else if (op == "-") {
        result = calc::subtract(a, b);
    } else if (op == "*") {
        result = calc::multiply(a, b);
    } else if (op == "/") {
        try {
            result = calc::divide(a, b);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Unknown operator: " << op << std::endl;
        return 1;
    }

    std::cout << result << std::endl;
    return 0;
}
```

**File: calculator/tests/CMakeLists.txt**
```cmake
find_package(PkgConfig QUIET)

# Option 1: Use GoogleTest if available
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(test_calc test_calculator.cpp)
    target_link_libraries(test_calc PRIVATE calclib GTest::gtest GTest::gtest_main)
    add_test(NAME test_calc COMMAND test_calc)
else()
    # Fallback: Simple test executable
    add_executable(test_calc test_calculator.cpp)
    target_link_libraries(test_calc PRIVATE calclib)
endif()
```

**File: calculator/tests/test_calculator.cpp**
```cpp
#include "calculator/math.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>

bool test_add() {
    double result = calc::add(2.0, 3.0);
    bool success = std::abs(result - 5.0) < 1e-10;
    std::cout << "add(2, 3) = " << result << " (expected: 5) " 
              << (success ? "PASS" : "FAIL") << std::endl;
    return success;
}

bool test_divide_by_zero() {
    bool exception_thrown = false;
    try {
        calc::divide(5.0, 0.0);
    } catch (const std::domain_error&) {
        exception_thrown = true;
    }
    std::cout << "divide by zero throws exception: " 
              << (exception_thrown ? "PASS" : "FAIL") << std::endl;
    return exception_thrown;
}

int main() {
    int failed = 0;
    
    if (!test_add()) failed++;
    if (!test_divide_by_zero()) failed++;
    
    std::cout << "\nTests run: 2, Failed: " << failed << std::endl;
    return failed > 0 ? 1 : 0;
}
```

**Build and Test Instructions:**
```bash
cd calculator
mkdir -p build
cd build
cmake .. -DBUILD_TESTS=ON
make
./calc 10 + 5
./test_calc
ctest
```

Expected outputs:
- `./calc 10 + 5` → `15`
- `./test_calc` → Shows test results
- `ctest` → Runs tests through CTest

## Advanced CMake Project

### Exercise 4: Cross-Platform Library with External Dependencies

Create a project that uses external libraries and supports cross-compilation:

```
cross_lib/
├── CMakeLists.txt
├── include/
│   └── cross_lib/
│       └── processor.hpp
├── src/
│   ├── CMakeLists.txt
│   └── processor.cpp
├── examples/
│   ├── CMakeLists.txt
│   └── example_usage.cpp
├── cmake/
│   ├── FindJSON.cmake
│   └── toolchain_arm.cmake
```

**File: cross_lib/CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.15)
project(CrossLib VERSION 1.0.0 DESCRIPTION "Cross-platform processing library")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(ENABLE_JSON "Enable JSON support" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

# Detect system
message(STATUS "Building for: ${CMAKE_SYSTEM_NAME} on ${CMAKE_SYSTEM_PROCESSOR}")

# Add source directory
add_subdirectory(src)

# Add examples if enabled
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CrossLibConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/CrossLibConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CrossLibConfig.cmake"
    INSTALL_DESTINATION lib/cmake/CrossLib
)

# Installation
include(GNUInstallDirs)
install(
    EXPORT CrossLibTargets
    FILE CrossLibTargets.cmake
    NAMESPACE CrossLib::
    DESTINATION lib/cmake/CrossLib
)

install(
    FILES 
        "${CMAKE_CURRENT_BINARY_DIR}/CrossLibConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/CrossLibConfigVersion.cmake"
    DESTINATION lib/cmake/CrossLib
)
```

**File: cross_lib/cmake/CrossLibConfig.cmake.in**
```cmake
@PACKAGE_INIT@

include("${CMAKE_CURRENT_LIST_DIR}/CrossLibTargets.cmake")

check_required_components(CrossLib)
```

**File: cross_lib/src/CMakeLists.txt**
```cmake
# Create library
add_library(crosslib
    processor.cpp
)

# Platform-specific definitions
if(WIN32)
    target_compile_definitions(crosslib PRIVATE PLATFORM_WIN)
elseif(APPLE)
    target_compile_definitions(crosslib PRIVATE PLATFORM_MACOS)
elseif(UNIX)
    target_compile_definitions(crosslib PRIVATE PLATFORM_LINUX)
endif()

# Include directories
target_include_directories(crosslib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

# JSON support
if(ENABLE_JSON)
    # Look for JSON library
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(JSON json-c QUIET)
    endif()
    
    if(NOT JSON_FOUND)
        # Fallback to header-only JSON library
        find_path(JSON_INCLUDE_DIR json/json.hpp PATHS external/)
        if(JSON_INCLUDE_DIR)
            target_include_directories(crosslib PRIVATE ${JSON_INCLUDE_DIR})
            target_compile_definitions(crosslib PRIVATE HAS_JSON_SUPPORT)
        endif()
    else()
        target_link_libraries(crosslib PRIVATE ${JSON_LIBRARIES})
        target_include_directories(crosslib PRIVATE ${JSON_INCLUDE_DIRS})
        target_compile_definitions(crosslib PRIVATE HAS_JSON_SUPPORT)
    endif()
endif()

# Export targets
set_target_properties(crosslib PROPERTIES
    EXPORT_NAME CrossLib
)

install(TARGETS crosslib
    EXPORT CrossLibTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

**File: cross_lib/include/cross_lib/processor.hpp**
```cpp
#ifndef CROSS_LIB_PROCESSOR_HPP
#define CROSS_LIB_PROCESSOR_HPP

#include <string>
#include <vector>

namespace cross_lib {

class DataProcessor {
public:
    DataProcessor();
    ~DataProcessor();
    
    void processData(const std::vector<double>& input, std::vector<double>& output);
    std::string getPlatformInfo() const;
    
#ifdef HAS_JSON_SUPPORT
    std::string serializeToJson(const std::vector<double>& data);
    std::vector<double> deserializeFromJson(const std::string& jsonStr);
#endif
    
private:
    int m_processorId;
};

} // namespace cross_lib

#endif
```

**File: cross_lib/src/processor.cpp**
```cpp
#include "cross_lib/processor.hpp"
#include <iostream>
#include <algorithm>

#ifdef PLATFORM_WIN
#include <windows.h>
#elif defined(PLATFORM_MACOS) || defined(PLATFORM_LINUX)
#include <unistd.h>
#endif

#ifdef HAS_JSON_SUPPORT
#include <json/json.h>  // This would be from json-c or similar
#endif

namespace cross_lib {

DataProcessor::DataProcessor() {
#ifdef PLATFORM_WIN
    m_processorId = GetCurrentProcessId();
#elif defined(PLATFORM_MACOS) || defined(PLATFORM_LINUX)
    m_processorId = getpid();
#else
    m_processorId = -1;
#endif
}

DataProcessor::~DataProcessor() {
    // Cleanup if needed
}

void DataProcessor::processData(const std::vector<double>& input, std::vector<double>& output) {
    output.resize(input.size());
    
    // Simple transformation: multiply by 2 and add processor ID
    std::transform(input.begin(), input.end(), output.begin(),
                   [this](double val) { return val * 2.0 + m_processorId; });
}

std::string DataProcessor::getPlatformInfo() const {
#ifdef PLATFORM_WIN
    return "Windows";
#elif defined(PLATFORM_MACOS)
    return "macOS";
#elif defined(PLATFORM_LINUX)
    return "Linux";
#else
    return "Unknown";
#endif
}

#ifdef HAS_JSON_SUPPORT
std::string DataProcessor::serializeToJson(const std::vector<double>& data) {
    // Simplified JSON serialization
    std::string result = "[";
    for (size_t i = 0; i < data.size(); ++i) {
        if (i > 0) result += ",";
        result += std::to_string(data[i]);
    }
    result += "]";
    return result;
}

std::vector<double> DataProcessor::deserializeFromJson(const std::string& jsonStr) {
    // Simplified JSON deserialization (in practice, use a proper JSON parser)
    std::vector<double> result;
    // This is a simplified implementation for demonstration
    // A real implementation would use a proper JSON parsing library
    return result;
}
#endif

} // namespace cross_lib
```

## Using Make Directly

### Exercise 5: Writing a Complex Makefile

For situations where you need to use Make directly, here's a comprehensive example:

**File: advanced_make_example/Makefile**
```makefile
# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -std=c11 -O2
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
LDFLAGS =
LDLIBS =

# Directory settings
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin
TESTDIR = test

# Find sources
SOURCES_C := $(wildcard $(SRCDIR)/*.c)
SOURCES_CPP := $(wildcard $(SRCDIR)/*.cpp)
SOURCES := $(SOURCES_C) $(SOURCES_CPP)

# Generate object file names
OBJECTS := $(SOURCES_C:$(SRCDIR)/%.c=$(OBJDIR)/%.o) \
           $(SOURCES_CPP:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# Define targets
TARGET = $(BINDIR)/myapp
TEST_TARGET = $(BINDIR)/test_runner

# Default target
.DEFAULT_GOAL := all

# Main target
all: $(TARGET)

# Create directories
$(OBJDIR):
	@mkdir -p $@

$(BINDIR):
	@mkdir -p $@

# Compile C files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "CC $<"
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# Compile C++ files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	@echo "CXX $<"
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

# Link main executable
$(TARGET): $(OBJECTS) | $(BINDIR)
	@echo "LINK $@"
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Test target
TEST_SOURCES := $(wildcard $(TESTDIR)/*.cpp)
TEST_OBJECTS := $(TEST_SOURCES:$(TESTDIR)/%.cpp=$(OBJDIR)/test_%.o)

$(OBJDIR)/test_%.o: $(TESTDIR)/%.cpp | $(OBJDIR)
	@echo "CXX $< (test)"
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -I$(TESTDIR) -c $< -o $@

$(TEST_TARGET): $(OBJECTS) $(TEST_OBJECTS) | $(BINDIR)
	@echo "LINK $@ (test)"
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

test: $(TEST_TARGET)
	@echo "Running tests..."
	./$(TEST_TARGET)

# Phony targets
.PHONY: all clean test debug release help

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(OBJDIR) $(BINDIR)

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: CXXFLAGS += -g -DDEBUG
debug: all

# Release build
release: CFLAGS += -O3 -DNDEBUG
release: CXXFLAGS += -O3 -DNDEBUG
release: all

# Install target
PREFIX ?= /usr/local
install: $(TARGET)
	@echo "Installing to $(PREFIX)/bin"
	@install -d $(PREFIX)/bin
	@install $(TARGET) $(PREFIX)/bin/

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build the main executable (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  debug     - Build with debug symbols"
	@echo "  release   - Build optimized release version"
	@echo "  test      - Build and run tests"
	@echo "  install   - Install the executable to PREFIX (default: /usr/local)"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  PREFIX    - Installation prefix (default: /usr/local)"

# Print variables for debugging
print-%:
	@echo $* = $($*)
```

**File: advanced_make_example/src/main.c**
```c
#include <stdio.h>

// Forward declaration
void utility_function();

int main() {
    printf("Hello from Make-built application!\n");
    utility_function();
    return 0;
}
```

**File: advanced_make_example/src/utils.c**
```c
#include <stdio.h>

void utility_function() {
    printf("Utility function called\n");
}
```

**File: advanced_make_example/include/utils.h**
```c
#ifndef UTILS_H
#define UTILS_H

void utility_function();

#endif
```

**File: advanced_make_example/test/test_main.cpp**
```cpp
#include "../include/utils.h"
#include <iostream>

int main() {
    std::cout << "Running tests..." << std::endl;
    
    // Call the utility function to test it
    utility_function();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
```

**Build Instructions:**
```bash
cd advanced_make_example
make
./bin/myapp
make test
make debug
make release
```

## Integration Examples

### Exercise 6: CMake with Custom Makefile Integration

Sometimes you might want to integrate CMake with custom Makefiles for specific tasks:

**File: integration_example/CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.15)
project(IntegrationExample)

set(CMAKE_CXX_STANDARD 17)

# Add executable
add_executable(app src/main.cpp)

# Add custom target that calls Make
add_custom_target(run-make
    COMMAND ${CMAKE_MAKE_PROGRAM} -f ${CMAKE_CURRENT_SOURCE_DIR}/custom.mk
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running custom Makefile"
)

# Add custom target for documentation
find_program(DOXYGEN_EXECUTABLE doxygen)
if(DOXYGEN_EXECUTABLE)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY
    )
    
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Add custom target for formatting
find_program(CLANG_FORMAT_EXECUTABLE clang-format)
if(CLANG_FORMAT_EXECUTABLE)
    file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h" "include/*.hpp")
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXECUTABLE} -i ${SOURCES}
        COMMENT "Formatting source files"
    )
endif()
```

**File: integration_example/custom.mk**
```makefile
# Custom Makefile that can be called from CMake
.PHONY: custom-task validate setup

custom-task:
	@echo "Running custom task from standalone Makefile"
	@echo "Current directory: $(PWD)"

validate:
	@echo "Validating project structure..."
	@test -d src || (echo "Error: src directory missing" && exit 1)
	@test -d include || (echo "Error: include directory missing" && exit 1)
	@echo "Validation passed!"

setup:
	@echo "Setting up development environment..."
	@if [ ! -d build ]; then mkdir build; fi
	@echo "Setup complete!"
```

**File: integration_example/src/main.cpp**
```cpp
#include <iostream>

int main() {
    std::cout << "Integrated CMake + Make example" << std::endl;
    return 0;
}
```

**File: integration_example/Doxyfile.in**
```
PROJECT_NAME = "Integration Example"
OUTPUT_DIRECTORY = @CMAKE_CURRENT_BINARY_DIR@/docs
INPUT = @CMAKE_CURRENT_SOURCE_DIR@/src @CMAKE_CURRENT_SOURCE_DIR@/include
GENERATE_HTML = YES
GENERATE_LATEX = NO
```

**Build and Run Instructions:**
```bash
cd integration_example
mkdir -p build
cd build
cmake ..
make
make run-make
make validate  # This would run the validation from custom.mk
```

## Troubleshooting Exercises

### Exercise 7: Debugging Common CMake Issues

Create a project with common mistakes to practice debugging:

**File: troubleshooting/CMakeLists.txt** (with intentional errors)
```cmake
cmake_minimum_required(VERSION 3.10)
project(TroubleShoot VERSION 1.0.0)

# Intentional error: using old-style commands
set(SOURCES src/main.cpp src/helper.cpp)
add_executable(trouble ${SOURCES})

# Another issue: missing include directory
target_link_libraries(trouble helper_lib)  # helper_lib doesn't exist

# Yet another issue: wrong path
include_directories(/nonexistent/path)  # This path doesn't exist
```

**Exercise Instructions:**
1. Try to build this project and identify the errors
2. Fix the issues using modern CMake practices:
   - Use `target_sources()` instead of `set()` + variable
   - Remove references to non-existent libraries
   - Fix invalid paths

**Corrected Version:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(TroubleShoot VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)

add_executable(trouble 
    src/main.cpp
    src/helper.cpp
)

# Only link libraries that actually exist
# target_link_libraries(trouble existing_lib)  # Uncomment when you have the lib
```

### Exercise 8: Performance Optimization

Learn to optimize build times by understanding dependency chains:

**File: optimization_example/CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.15)
project(OptimizationExample)

set(CMAKE_CXX_STANDARD 17)

# Create a library with many headers
add_library(biglib
    src/biglib.cpp
)

target_include_directories(biglib PUBLIC
    include/
)

# Create executable that only uses part of the library
add_executable(selective_user src/selective_user.cpp)
target_link_libraries(selective_user PRIVATE biglib)

# Create a header-only library for comparison
add_library(header_only INTERFACE)
target_include_directories(header_only INTERFACE include/)

# Demonstrate precompiled headers (if supported)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
    set_source_files_properties(src/biglib.cpp PROPERTIES
        CMAKE_PCH_HEADER include/biglib.hpp
    )
endif()
```

## Summary

This hands-on tutorial covered:

1. **Basic CMake usage** - Creating simple projects with executables and libraries
2. **Intermediate concepts** - Multi-directory projects, testing, installation
3. **Advanced topics** - Cross-platform development, external dependencies, packaging
4. **Pure Make usage** - Complex Makefiles for when CMake isn't appropriate
5. **Integration techniques** - Combining CMake and Make for complex workflows
6. **Troubleshooting** - Identifying and fixing common build issues

Practice these exercises to become proficient with CMake and Make. Remember that mastery comes with experience, so try adapting these examples to your own projects.