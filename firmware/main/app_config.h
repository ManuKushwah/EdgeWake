/*
 * app_config.h - task layout, priorities and buffer sizes.
 *
 * Core 0 (PRO_CPU): Wi-Fi, LwIP, MQTT, the audio capture task (the ring
 *                   buffer producer), telemetry, health, OTA.
 * Core 1 (APP_CPU): the KWS task alone (ring buffer consumer: VAD, MFCC,
 *                   CNN). Nothing networking-related can preempt it.
 *
 * ESP-IDF reference priorities (configMAX_PRIORITIES = 25): IPC 24,
 * Wi-Fi 23, esp_timer 22, LwIP 18, main 1, idle 0.
 */
#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "freertos/FreeRTOS.h"

#define EW_CORE_AUDIO       0
#define EW_CORE_KWS         1
#define EW_CORE_NET         0

/* Capture does ~20 us of work per 20 ms, so it can sit just under the
 * Wi-Fi/timer tasks. The DMA buffers (below) absorb any Wi-Fi burst. */
#define EW_PRIO_AUDIO       21
/* Core 1 has nothing else to run; the priority only matters relative to
 * unpinned tasks, which it must preempt. */
#define EW_PRIO_KWS         15
#define EW_PRIO_TELEMETRY   5
#define EW_PRIO_OTA         4
#define EW_PRIO_HEALTH      3

/* Stack sizes are in bytes on ESP-IDF. Health telemetry reports each task's
 * high-water mark so these can be trimmed from on-device data. */
#define EW_STACK_AUDIO      3072
#define EW_STACK_KWS        6144
#define EW_STACK_TELEMETRY  6144
#define EW_STACK_HEALTH     4096
#define EW_STACK_OTA        8192

/* I2S DMA: 6 descriptors x 320 frames (20 ms each). The driver keeps one in
 * flight, so 5 completed buffers = 100 ms can wait for the capture task. */
#define EW_I2S_DMA_DESC_NUM 6

/* Audio ring: 8192 samples = 512 ms of slack between capture and KWS. */
#define EW_AUDIO_RING_SAMPLES 8192

/* Load shedding in the KWS task: with this many hops (80 ms) still waiting
 * in the ring, skip due inferences until caught up, and never process more
 * than EW_KWS_MAX_HOPS_PER_WAKE hops without feeding the watchdog and
 * yielding so IDLE1 can run. Keeps a slow configuration from starving the
 * task watchdog into a reboot loop. */
#define EW_KWS_SHED_BACKLOG_HOPS  4
#define EW_KWS_MAX_HOPS_PER_WAKE  8

/* Detection events queued for telemetry; the KWS task never blocks on it. */
#define EW_EVENT_QUEUE_LEN  8

#endif /* APP_CONFIG_H */
