cmake_minimum_required(VERSION 3.24)

# Force real macOS SDK before project() so compiler detection uses it.
# Needed when building inside nix devenv where SDKROOT may point to
# nix's stripped apple-sdk that lacks Metal/MPS headers.
execute_process(
    COMMAND /usr/bin/xcrun --sdk macosx --show-sdk-path
    OUTPUT_VARIABLE REAL_MACOS_SDK
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(REAL_MACOS_SDK)
    set(CMAKE_OSX_SYSROOT "${REAL_MACOS_SDK}" CACHE STRING "" FORCE)
endif()

project(faiss-metal LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(OpenMP REQUIRED)
find_package(faiss REQUIRED)

# Fix faiss imported target: replace nix SDK framework paths with real SDK paths.
# The nix faiss package hardcodes nix apple-sdk paths for Accelerate.framework,
# which adds -F<nix-sdk>/Frameworks to compile flags. This causes Metal.framework
# and Foundation.framework headers to be found from the nix SDK (14.4) instead of
# the real macOS SDK, leading to header version conflicts.
if(REAL_MACOS_SDK AND TARGET faiss)
    get_target_property(_faiss_iface faiss INTERFACE_LINK_LIBRARIES)
    if(_faiss_iface)
        string(REGEX REPLACE
            "/nix/store/[^/]+/Platforms/MacOSX\\.platform/Developer/SDKs/MacOSX\\.sdk/System/Library/Frameworks/([A-Za-z]+)\\.framework"
            "${REAL_MACOS_SDK}/System/Library/Frameworks/\\1.framework"
            _faiss_iface_fixed "${_faiss_iface}")
        set_target_properties(faiss PROPERTIES INTERFACE_LINK_LIBRARIES "${_faiss_iface_fixed}")
    endif()
endif()

# --- Compile .metal shaders → .metallib ---
file(GLOB METAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.metal)

# Use /usr/bin/xcrun to bypass nix xcrun wrapper which lacks Metal Toolchain support
find_program(XCRUN_EXECUTABLE xcrun PATHS /usr/bin NO_DEFAULT_PATH)
if(NOT XCRUN_EXECUTABLE)
    find_program(XCRUN_EXECUTABLE xcrun)
endif()

# Compile each .metal to .air, then link into one .metallib
set(METAL_AIR_FILES "")
foreach(METAL_SRC ${METAL_SOURCES})
    get_filename_component(METAL_NAME ${METAL_SRC} NAME_WE)
    set(AIR_FILE ${CMAKE_BINARY_DIR}/${METAL_NAME}.air)
    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -std=metal3.1 -mmacos-version-min=15.0 -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders -c ${METAL_SRC} -o ${AIR_FILE}
        DEPENDS ${METAL_SRC}
        COMMENT "Compiling ${METAL_NAME}.metal → .air"
    )
    list(APPEND METAL_AIR_FILES ${AIR_FILE})
endforeach()

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/faiss_metal.metallib
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${METAL_AIR_FILES} -o ${CMAKE_BINARY_DIR}/faiss_metal.metallib
    DEPENDS ${METAL_AIR_FILES}
    COMMENT "Linking Metal shaders → faiss_metal.metallib"
)
add_custom_target(metal_shaders DEPENDS ${CMAKE_BINARY_DIR}/faiss_metal.metallib)

# --- Library ---
set(FAISS_METAL_SOURCES
    src/MetalContext.mm
    src/MetalDeviceCapabilities.mm
    src/StandardMetalResources.mm
    src/MetalL2Norm.mm
    src/MetalDistance.mm
    src/MetalSelect.mm
    src/MetalIndexFlat.mm
    src/MetalIndexIVFFlat.mm
)

add_library(faiss_metal STATIC ${FAISS_METAL_SOURCES})
add_dependencies(faiss_metal metal_shaders)

target_include_directories(faiss_metal
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
)

# Prioritize real SDK framework headers over nix apple-sdk frameworks.
# The nix faiss package links against nix SDK's Accelerate.framework which
# adds the nix SDK's framework directory to the search path. Without this,
# Metal.h and Foundation.h are found from the nix SDK (14.4) which is too
# old and causes header conflicts with the real macOS SDK.
if(REAL_MACOS_SDK)
    target_compile_options(faiss_metal PRIVATE
        -iframework ${REAL_MACOS_SDK}/System/Library/Frameworks)
endif()

target_link_libraries(faiss_metal
    PUBLIC
        faiss
    PRIVATE
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework Accelerate"
)

# Pass metallib path as compile definition
target_compile_definitions(faiss_metal PRIVATE
    FAISS_METAL_METALLIB_PATH="${CMAKE_BINARY_DIR}/faiss_metal.metallib"
)

# --- Tests ---
option(FAISS_METAL_BUILD_TESTS "Build tests" ON)
if(FAISS_METAL_BUILD_TESTS)
    enable_testing()

    add_executable(test_metal_flat tests/test_metal_flat.mm)
    target_link_libraries(test_metal_flat faiss_metal faiss
        "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
    add_test(NAME test_metal_flat COMMAND test_metal_flat)

    add_executable(test_metal_distance tests/test_metal_distance.mm)
    target_link_libraries(test_metal_distance faiss_metal faiss
        "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
    add_test(NAME test_metal_distance COMMAND test_metal_distance)

    add_executable(bench_metal_flat tests/bench_metal_flat.mm)
    target_link_libraries(bench_metal_flat faiss_metal faiss
        "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")

    add_executable(test_metal_ivfflat tests/test_metal_ivfflat.mm)
    target_link_libraries(test_metal_ivfflat faiss_metal faiss
        "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
    add_test(NAME test_metal_ivfflat COMMAND test_metal_ivfflat)
endif()
