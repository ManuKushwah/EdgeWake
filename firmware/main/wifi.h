#ifndef WIFI_H
#define WIFI_H

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

/* Starts the station and keeps reconnecting with exponential backoff.
 * Returns ESP_ERR_INVALID_STATE when no SSID is configured (offline mode). */
esp_err_t wifi_start(void);

bool wifi_wait_connected(uint32_t timeout_ms);
bool wifi_is_connected(void);
int wifi_rssi(void);

#endif /* WIFI_H */
