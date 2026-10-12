# Host build of the portable core: unit tests (ASan/UBSan + a TSan stress
# test for the lock-free ring buffer) and a shared library that the Python
# training pipeline loads with ctypes, so training features come from the
# exact C code that runs on the device.
cmake_minimum_required(VERSION 3.16)
project(edgewake_host C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../firmware/components/edgewake_core)
set(CORE_SRCS
    ${CORE_DIR}/src/ew_ringbuf.c
    ${CORE_DIR}/src/ew_fft.c
    ${CORE_DIR}/src/ew_mfcc.c
    ${CORE_DIR}/src/ew_vad.c
    ${CORE_DIR}/src/ew_kws.c)
set(WARN_FLAGS -Wall -Wextra -Werror -Wshadow)

find_package(Threads REQUIRED)

# --- core built with sanitizers, for the unit tests ---------------------------
add_library(ew_core_san STATIC ${CORE_SRCS})
target_include_directories(ew_core_san PUBLIC ${CORE_DIR}/include)
target_compile_options(ew_core_san PRIVATE ${WARN_FLAGS} -Wdouble-promotion)
target_compile_options(ew_core_san PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
target_link_options(ew_core_san PUBLIC -fsanitize=address,undefined)
target_link_libraries(ew_core_san PUBLIC m)

enable_testing()
foreach(t ringbuf fft mfcc vad kws)
    add_executable(test_${t} tests/test_${t}.c)
    target_compile_options(test_${t} PRIVATE ${WARN_FLAGS})
    target_link_libraries(test_${t} PRIVATE ew_core_san Threads::Threads)
    add_test(NAME ${t} COMMAND test_${t})
endforeach()

# --- ring buffer stress test under ThreadSanitizer (cannot mix with ASan) -----
add_executable(test_ringbuf_tsan tests/test_ringbuf.c ${CORE_DIR}/src/ew_ringbuf.c)
target_include_directories(test_ringbuf_tsan PRIVATE ${CORE_DIR}/include)
target_compile_options(test_ringbuf_tsan PRIVATE ${WARN_FLAGS} -fsanitize=thread -O1 -g)
target_link_options(test_ringbuf_tsan PRIVATE -fsanitize=thread)
target_link_libraries(test_ringbuf_tsan PRIVATE Threads::Threads)
# Older TSan runtimes abort under the high ASLR entropy of recent kernels;
# `setarch -R` disables randomisation for this one process.
add_test(NAME ringbuf_tsan COMMAND setarch ${CMAKE_HOST_SYSTEM_PROCESSOR} -R $<TARGET_FILE:test_ringbuf_tsan>)

# --- shared library for Python (ctypes); plain -O2, no sanitizers --------------
add_library(edgewake SHARED pyapi/ew_pyapi.c ${CORE_SRCS})
target_include_directories(edgewake PRIVATE ${CORE_DIR}/include)
target_compile_options(edgewake PRIVATE ${WARN_FLAGS} -O2)
target_link_libraries(edgewake PRIVATE m)
