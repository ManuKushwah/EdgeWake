/*
 * audio_capture.h - I2S microphone capture (producer side of the audio ring).
 */
#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"
#include "ew_ringbuf.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

typedef struct {
    uint32_t chunks;           /* 20 ms chunks pushed into the ring */
    uint32_t read_errors;      /* I2S reads that timed out / came up short */
    uint32_t dropped_samples;  /* ring overflow (consumer fell > 512 ms behind) */
    int32_t last_peak;         /* |peak| of the latest chunk, int16 units */
} audio_stats_t;

/* Initializes the static ring buffer. Call before any task uses it. */
void audio_ring_init(void);
ew_ringbuf_t *audio_ring(void);

/* Starts the capture task on core 0; it notifies `consumer` after each chunk. */
esp_err_t audio_capture_start(TaskHandle_t consumer);
TaskHandle_t audio_task_handle(void);

void audio_get_stats(audio_stats_t *out);

/* True if the microphone produced a non-constant signal within the last
 * second (catches a disconnected or dead mic in the OTA self-test). */
bool audio_capture_is_alive(void);

#endif /* AUDIO_CAPTURE_H */
