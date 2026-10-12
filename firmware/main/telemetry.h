/*
 * telemetry.h - MQTT over mutual TLS to AWS IoT Core.
 *
 * Topics (<thing> = CONFIG_EW_THING_NAME):
 *   edgewake/<thing>/detections  QoS 1  one JSON message per keyword
 *   edgewake/<thing>/health      QoS 0  periodic health snapshot
 *   edgewake/<thing>/status      QoS 1  {"online":true}; LWT {"online":false}
 *   edgewake/<thing>/cmd         (subscribed) {"ota_url":"https://..."},
 *                                {"vad":false}, {"threshold":0.85}
 */
#ifndef TELEMETRY_H
#define TELEMETRY_H

#include <stdbool.h>

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

/* Returns ESP_ERR_NOT_SUPPORTED when built without certificates or endpoint. */
esp_err_t telemetry_start(QueueHandle_t detection_events);
bool telemetry_connected(void);

#endif /* TELEMETRY_H */
