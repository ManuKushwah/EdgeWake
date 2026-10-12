/*
 * ew_port.h - tiny portability layer so the DSP core compiles unchanged for
 * the ESP32-S3 (ESP-IDF) and for host unit tests (gcc/clang on Linux).
 */
#ifndef EW_PORT_H
#define EW_PORT_H

#if defined(ESP_PLATFORM)
#include "esp_attr.h"
/* Place hot loops in internal IRAM: zero-wait-state, no flash-cache misses. */
#define EW_IRAM IRAM_ATTR
#else
#define EW_IRAM
#endif

#define EW_LIKELY(x)   __builtin_expect(!!(x), 1)
#define EW_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define EW_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

#endif /* EW_PORT_H */
