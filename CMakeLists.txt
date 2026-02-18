cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME ann)

# ========================================
# 1. Rust DiskANN library (staticlib via cargo)
# ========================================

option(ENABLE_RUST_DISKANN "Build and link Rust DiskANN library" ON)

if(ENABLE_RUST_DISKANN)
    find_program(CARGO_EXECUTABLE cargo)

    if(CARGO_EXECUTABLE)
        message(STATUS "Rust toolchain found: ${CARGO_EXECUTABLE}")

        # Determine build type
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(RUST_BUILD_TYPE "debug")
            set(RUST_BUILD_FLAG "")
        else()
            set(RUST_BUILD_TYPE "release")
            set(RUST_BUILD_FLAG "--release")
        endif()

        # Detect Rust target
        set(RUST_TARGET "")
        if("${OS_NAME}" STREQUAL "linux")
            if("${OS_ARCH}" STREQUAL "arm64" OR "${CMAKE_CXX_COMPILER}" MATCHES "aarch64")
                set(RUST_TARGET "aarch64-unknown-linux-gnu")
            else()
                set(RUST_TARGET "x86_64-unknown-linux-gnu")
            endif()
        elseif("${OS_NAME}" STREQUAL "osx" OR APPLE)
            if("${OSX_BUILD_ARCH}" STREQUAL "arm64" OR "${OS_ARCH}" STREQUAL "arm64"
               OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
                set(RUST_TARGET "aarch64-apple-darwin")
            else()
                set(RUST_TARGET "x86_64-apple-darwin")
            endif()
        elseif(WIN32)
            if("${OS_ARCH}" STREQUAL "arm64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "ARM64")
                set(RUST_TARGET "aarch64-pc-windows-msvc")
            else()
                set(RUST_TARGET "x86_64-pc-windows-msvc")
            endif()
        endif()

        if(NOT "${RUST_TARGET}" STREQUAL "")
            set(RUST_TARGET_FLAG "--target=${RUST_TARGET}")
            set(RUST_TARGET_DIR "${RUST_TARGET}/")
            message(STATUS "Rust target: ${RUST_TARGET}")
        else()
            set(RUST_TARGET_FLAG "")
            set(RUST_TARGET_DIR "")
        endif()

        # Platform-specific library name
        if(WIN32)
            set(RUST_LIB_NAME "diskann_rust.lib")
        else()
            set(RUST_LIB_NAME "libdiskann_rust.a")
        endif()

        set(RUST_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rust_lib)
        set(RUST_LIB_PATH ${RUST_LIB_DIR}/target/${RUST_TARGET_DIR}${RUST_BUILD_TYPE}/${RUST_LIB_NAME})

        # Custom command to build Rust library
        add_custom_command(
            OUTPUT ${RUST_LIB_PATH}
            COMMAND ${CARGO_EXECUTABLE} build ${RUST_BUILD_FLAG} ${RUST_TARGET_FLAG}
            WORKING_DIRECTORY ${RUST_LIB_DIR}
            COMMENT "Building Rust DiskANN library..."
            DEPENDS
                ${RUST_LIB_DIR}/Cargo.toml
                ${RUST_LIB_DIR}/src/lib.rs
                ${RUST_LIB_DIR}/src/ffi.rs
                ${RUST_LIB_DIR}/src/index_manager.rs
                ${RUST_LIB_DIR}/src/provider.rs
                ${RUST_LIB_DIR}/src/runtime.rs
                ${RUST_LIB_DIR}/src/file_format.rs
                ${RUST_LIB_DIR}/src/disk_provider.rs
                ${RUST_LIB_DIR}/src/distance.rs
                ${RUST_LIB_DIR}/src/metal_ffi.rs
                ${RUST_LIB_DIR}/src/streaming_build.rs
        )

        add_custom_target(diskann_rust_build DEPENDS ${RUST_LIB_PATH})

        add_library(diskann_rust STATIC IMPORTED GLOBAL)
        set_target_properties(diskann_rust PROPERTIES
            IMPORTED_LOCATION ${RUST_LIB_PATH}
        )
        add_dependencies(diskann_rust diskann_rust_build)

        set(DISKANN_RUST_AVAILABLE ON)
        message(STATUS "Rust DiskANN: ENABLED")
    else()
        message(STATUS "Rust toolchain not found - DiskANN: DISABLED")
        set(DISKANN_RUST_AVAILABLE OFF)
    endif()
else()
    message(STATUS "Rust DiskANN: DISABLED (ENABLE_RUST_DISKANN=OFF)")
    set(DISKANN_RUST_AVAILABLE OFF)
endif()

# ========================================
# 2. Optional FAISS library
# ========================================

option(ENABLE_FAISS "Build with FAISS support" ON)
set(FAISS_AVAILABLE OFF)

if(ENABLE_FAISS)
    # OpenMP must be found before faiss so OpenMP::OpenMP_CXX target exists
    # when cmake resolves faiss's interface link libraries.
    find_package(OpenMP QUIET)
    find_package(faiss QUIET)
    if(faiss_FOUND)
        set(FAISS_AVAILABLE ON)
        message(STATUS "FAISS: FOUND")
    else()
        message(STATUS "FAISS: NOT FOUND (faiss_* functions disabled)")
    endif()
endif()

# ========================================
# 3. Optional Metal GPU backend (macOS only, requires FAISS)
# ========================================

set(FAISS_METAL_AVAILABLE OFF)
if(FAISS_AVAILABLE AND APPLE)
    find_library(METAL_FW Metal)
    find_library(MPS_FW MetalPerformanceShaders)
    if(METAL_FW AND MPS_FW)
        set(FAISS_METAL_AVAILABLE ON)
        enable_language(OBJCXX)

        # Force real macOS SDK for Metal headers (nix apple-sdk lacks them)
        execute_process(
            COMMAND /usr/bin/xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE REAL_MACOS_SDK
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        # Build faiss-metal from in-tree source
        set(FAISS_METAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/faiss-metal")
        set(FAISS_METAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        add_subdirectory(${FAISS_METAL_DIR})
        # Fix include dirs for install export (source-dir paths not allowed)
        set_target_properties(faiss_metal PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${FAISS_METAL_DIR}/include>")
        message(STATUS "FAISS Metal GPU: ENABLED")
    else()
        message(STATUS "FAISS Metal GPU: DISABLED (Metal frameworks not found)")
    endif()
else()
    if(NOT FAISS_AVAILABLE)
        message(STATUS "FAISS Metal GPU: DISABLED (FAISS not available)")
    elseif(NOT APPLE)
        message(STATUS "FAISS Metal GPU: DISABLED (not macOS)")
    endif()
endif()

# ========================================
# 4. C++ Extension sources
# ========================================

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)

# Core sources (always compiled)
set(EXTENSION_SOURCES
    src/ann_extension.cpp
    src/ann_search.cpp
    src/ann_optimizer.cpp
    src/diskann_functions.cpp
    src/diskann_index.cpp
    src/rust_ffi.cpp
    src/ann_list.cpp
)

# FAISS sources (conditionally compiled via #ifdef FAISS_AVAILABLE in each file)
list(APPEND EXTENSION_SOURCES
    src/faiss_index.cpp
    src/faiss_fn_gpu.cpp
    src/faiss_vector_utils.cpp
    src/gpu_backend_cpu.cpp
    src/metal_diskann_stub.cpp
)

# Metal GPU backend (Objective-C++)
if(FAISS_METAL_AVAILABLE)
    list(APPEND EXTENSION_SOURCES src/gpu_backend_metal.mm src/metal_diskann_bridge.mm)
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# ========================================
# 5. Linking
# ========================================

# Link Rust library (DiskANN)
if(DISKANN_RUST_AVAILABLE)
    add_dependencies(${EXTENSION_NAME} diskann_rust_build)
    add_dependencies(${LOADABLE_EXTENSION_NAME} diskann_rust_build)

    target_link_libraries(${EXTENSION_NAME} ${RUST_LIB_PATH})
    target_link_libraries(${LOADABLE_EXTENSION_NAME} ${RUST_LIB_PATH})

    # Platform-specific system libraries required by Rust
    if(APPLE)
        target_link_libraries(${EXTENSION_NAME}
            "-framework Security" "-framework CoreFoundation" "-framework SystemConfiguration")
        target_link_libraries(${LOADABLE_EXTENSION_NAME}
            "-framework Security" "-framework CoreFoundation" "-framework SystemConfiguration")
    elseif(WIN32)
        target_link_libraries(${EXTENSION_NAME} ws2_32 userenv bcrypt ntdll)
        target_link_libraries(${LOADABLE_EXTENSION_NAME} ws2_32 userenv bcrypt ntdll)
    elseif(UNIX)
        target_link_libraries(${EXTENSION_NAME} dl pthread m)
        target_link_libraries(${LOADABLE_EXTENSION_NAME} dl pthread m)
    endif()

    target_compile_definitions(${EXTENSION_NAME} PRIVATE DISKANN_RUST_AVAILABLE=1)
    target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE DISKANN_RUST_AVAILABLE=1)
endif()

# Link FAISS library
if(FAISS_AVAILABLE)
    target_link_libraries(${EXTENSION_NAME} faiss)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} faiss)

    target_compile_definitions(${EXTENSION_NAME} PRIVATE FAISS_AVAILABLE=1)
    target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE FAISS_AVAILABLE=1)

    # FAISS headers include omp.h — propagate OpenMP include directory
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${EXTENSION_NAME} OpenMP::OpenMP_CXX)
        target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenMP::OpenMP_CXX)
    endif()
endif()

# Link Metal GPU backend
if(FAISS_METAL_AVAILABLE)
    target_link_libraries(${EXTENSION_NAME} faiss_metal
        "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
    target_link_libraries(${LOADABLE_EXTENSION_NAME} faiss_metal
        "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")

    target_compile_definitions(${EXTENSION_NAME} PRIVATE FAISS_METAL_ENABLED=1
        FAISS_METAL_METALLIB_PATH="${CMAKE_BINARY_DIR}/faiss_metal.metallib")
    target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE FAISS_METAL_ENABLED=1
        FAISS_METAL_METALLIB_PATH="${CMAKE_BINARY_DIR}/faiss_metal.metallib")

    # Prioritize real SDK framework headers over nix apple-sdk
    if(REAL_MACOS_SDK)
        target_compile_options(${EXTENSION_NAME} PRIVATE
            -iframework ${REAL_MACOS_SDK}/System/Library/Frameworks)
        target_compile_options(${LOADABLE_EXTENSION_NAME} PRIVATE
            -iframework ${REAL_MACOS_SDK}/System/Library/Frameworks)
    endif()
endif()

set(_INSTALL_TARGETS ${EXTENSION_NAME})
if(FAISS_METAL_AVAILABLE)
    list(APPEND _INSTALL_TARGETS faiss_metal)
endif()
install(
    TARGETS ${_INSTALL_TARGETS}
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
