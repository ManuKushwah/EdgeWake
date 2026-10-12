#include "ota.h"

#include <stdatomic.h>
#include <string.h>

#include "app_config.h"
#include "esp_app_desc.h"
#include "esp_crt_bundle.h"
#include "esp_http_client.h"
#include "esp_https_ota.h"
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "ota";

static bool s_pending_verify;
static const char *s_state = "unknown";
static char s_url[256];
static atomic_bool s_busy;
static StaticTask_t s_tcb;
static StackType_t s_stack[EW_STACK_OTA];
static TaskHandle_t s_task;

static void ota_task(void *arg);

void ota_init(void)
{
    const esp_partition_t *running = esp_ota_get_running_partition();
    esp_ota_img_states_t state;
    if (esp_ota_get_state_partition(running, &state) == ESP_OK) {
        switch (state) {
        case ESP_OTA_IMG_PENDING_VERIFY:
            s_pending_verify = true;
            s_state = "pending_verify";
            break;
        case ESP_OTA_IMG_VALID:
            s_state = "valid";
            break;
        default:
            s_state = "other";
            break;
        }
    } else {
        s_state = "factory";  /* flashed over serial: no OTA state yet */
    }
    ESP_LOGI(TAG, "running %s @ 0x%06lx, fw %s, state %s", running->label,
             (unsigned long)running->address, esp_app_get_description()->version, s_state);

    /* A persistent worker that sleeps until asked: re-creating a task in the
     * same static TCB/stack after vTaskDelete() would race the idle task's
     * cleanup of the old one. */
    s_task = xTaskCreateStaticPinnedToCore(ota_task, "ota", EW_STACK_OTA, NULL, EW_PRIO_OTA,
                                           s_stack, &s_tcb, EW_CORE_NET);
}

bool ota_pending_verify(void)
{
    return s_pending_verify;
}

void ota_confirm(bool ok, const char *reason)
{
    if (!s_pending_verify) {
        return;
    }
    if (ok) {
        ESP_LOGI(TAG, "self-test passed: marking image valid");
        esp_ota_mark_app_valid_cancel_rollback();
        s_pending_verify = false;
        s_state = "valid";
    } else {
        ESP_LOGE(TAG, "self-test FAILED (%s): rolling back", reason);
        /* Reboots on success. It returns only when otadata has no valid
         * record of another slot (e.g. a board whose first image was flashed
         * over USB and never booted through the OTA path). */
        const esp_err_t err = esp_ota_mark_app_invalid_rollback_and_reboot();
        /* Reset anyway with the image still PENDING_VERIFY: the bootloader
         * then marks it ABORTED and scans the slots for any other valid
         * image; if there is none it boots this one again (now ABORTED, so no
         * second self-test). Either way the device is never left without a
         * bootable app. Verified in QEMU, see docs/ARCHITECTURE.md. */
        ESP_LOGE(TAG, "rollback API refused (%s): resetting into the bootloader's fallback",
                 esp_err_to_name(err));
        esp_restart();
    }
}

static esp_err_t run_update(void)
{
    ESP_LOGI(TAG, "downloading %s", s_url);
    esp_http_client_config_t http = {
        .url = s_url,
        .crt_bundle_attach = esp_crt_bundle_attach,  /* HTTPS only, CA-verified */
        .timeout_ms = 15000,
        .keep_alive_enable = true,
    };
    esp_https_ota_config_t cfg = {.http_config = &http};
    esp_https_ota_handle_t handle = NULL;

    esp_err_t err = esp_https_ota_begin(&cfg, &handle);
    if (err == ESP_OK) {
        esp_app_desc_t incoming;
        err = esp_https_ota_get_img_desc(handle, &incoming);
        if (err == ESP_OK && strcmp(incoming.version, esp_app_get_description()->version) == 0) {
            ESP_LOGW(TAG, "image version %s already running", incoming.version);
            err = ESP_ERR_INVALID_VERSION;
        }
        while (err == ESP_OK && (err = esp_https_ota_perform(handle)) == ESP_ERR_HTTPS_OTA_IN_PROGRESS) {
            /* Each call writes one chunk into the inactive slot. */
        }
        if (err == ESP_OK && esp_https_ota_is_complete_data_received(handle)) {
            err = esp_https_ota_finish(handle);  /* verifies image, sets boot slot */
            handle = NULL;
        } else if (err == ESP_OK) {
            err = ESP_ERR_INVALID_SIZE;
        }
        if (handle != NULL) {
            esp_https_ota_abort(handle);
        }
    }

    return err;
}

static void ota_task(void *arg)
{
    (void)arg;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        const esp_err_t err = run_update();
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "update written; rebooting into the new image");
            vTaskDelay(pdMS_TO_TICKS(500));
            esp_restart();
        }
        ESP_LOGE(TAG, "update failed: %s", esp_err_to_name(err));
        atomic_store(&s_busy, false);
    }
}

esp_err_t ota_start_update(const char *url)
{
    if (strncmp(url, "https://", 8) != 0 || strlen(url) >= sizeof(s_url)) {
        return ESP_ERR_INVALID_ARG;
    }
    if (atomic_exchange(&s_busy, true)) {
        return ESP_ERR_INVALID_STATE;
    }
    strlcpy(s_url, url, sizeof(s_url));
    xTaskNotifyGive(s_task);  /* also orders the s_url write before the wake-up */
    return ESP_OK;
}

const char *ota_running_partition(void)
{
    return esp_ota_get_running_partition()->label;
}

const char *ota_state_str(void)
{
    return s_state;
}
