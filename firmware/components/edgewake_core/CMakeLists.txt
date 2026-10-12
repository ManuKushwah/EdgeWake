# Portable DSP / KWS core. Built as an ESP-IDF component for the firmware and
# directly by host/CMakeLists.txt for unit tests and the Python bindings.
idf_component_register(
    SRCS "src/ew_ringbuf.c"
         "src/ew_fft.c"
         "src/ew_mfcc.c"
         "src/ew_vad.c"
         "src/ew_kws.c"
    INCLUDE_DIRS "include")

# Single-precision FPU on the S3: flag any accidental float -> double math.
target_compile_options(${COMPONENT_LIB} PRIVATE -Wdouble-promotion -O2)
