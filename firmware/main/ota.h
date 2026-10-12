/*
 * ota.h - A/B over-the-air updates with automatic rollback.
 *
 * Flow:
 *   1. An MQTT command carries an HTTPS URL; ota_start_update() streams the
 *      image into the *inactive* slot (ota_0 <-> ota_1) and validates it.
 *   2. The new slot is set as boot partition and the chip reboots. The
 *      bootloader marks the new image PENDING_VERIFY.
 *   3. On that first boot app_main runs the self-test and calls ota_confirm():
 *      pass -> image marked VALID (rollback cancelled),
 *      fail -> marked INVALID and the chip reboots into the old slot.
 *   4. If the new image crashes or trips a watchdog before confirming, the
 *      bootloader sees an unconfirmed image on the next boot and rolls back.
 */
#ifndef OTA_H
#define OTA_H

#include <stdbool.h>

#include "esp_err.h"

/* Logs the running slot and detects a PENDING_VERIFY first boot. */
void ota_init(void);
bool ota_pending_verify(void);

/* Marks the running image valid, or invalid (which reboots into the
 * previous image and never returns). */
void ota_confirm(bool ok, const char *reason);

/* Starts the download task; returns ESP_ERR_INVALID_STATE if one is running. */
esp_err_t ota_start_update(const char *url);

const char *ota_running_partition(void);
const char *ota_state_str(void);

#endif /* OTA_H */
