/*
 * kws_task.h - the keyword-spotting task (consumer side of the audio ring).
 */
#ifndef KWS_TASK_H
#define KWS_TASK_H

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"
#include "ew_kws.h"
#include "ew_ringbuf.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "inference.h"

/* Posted to telemetry for every detection (never blocks the KWS task). */
typedef struct {
    uint32_t seq;
    uint32_t timestamp_ms;
    uint32_t hop_to_action_us;  /* hop dequeued -> GPIO action, measured on-device */
    float score;
    uint8_t class_index;
} kws_event_t;

typedef struct {
    ew_kws_stats_t engine;      /* hops, inference slots, VAD skips, detections */
    uint64_t busy_us;           /* total time spent processing hops */
    uint32_t hop_us_max;
    uint32_t infer_us_last;
    uint32_t infer_us_max;
    uint64_t infer_us_sum;
    uint32_t infer_count;
    uint32_t events_dropped;    /* telemetry queue was full */
    bool vad_enabled;
    float threshold;
} kws_stats_t;

/* Initializes the engine with the model's quantization parameters and
 * starts the task pinned to core 1. */
esp_err_t kws_task_start(const inference_info_t *info, ew_ringbuf_t *ring, QueueHandle_t events);

TaskHandle_t kws_task_handle(void);

void kws_get_stats(kws_stats_t *out);

/* Runtime tuning from other tasks (e.g. MQTT commands). Applied by the KWS
 * task itself at the top of its loop, so the engine has a single writer. */
void kws_request_vad(bool enabled);
void kws_request_threshold(float threshold);

#endif /* KWS_TASK_H */
