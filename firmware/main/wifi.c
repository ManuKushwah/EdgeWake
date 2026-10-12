#include "wifi.h"

#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "sdkconfig.h"

static const char *TAG = "wifi";

#define CONNECTED_BIT BIT0

static StaticEventGroup_t s_group_mem;
static EventGroupHandle_t s_group;
static esp_timer_handle_t s_retry_timer;
static uint32_t s_attempt;

static void retry_cb(void *arg)
{
    (void)arg;
    esp_wifi_connect();
}

static void on_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_group, CONNECTED_BIT);
        /* 1 s, 2 s, 4 s ... capped at 32 s, so a dead AP doesn't spin the radio. */
        const uint32_t shift = s_attempt < 5 ? s_attempt : 5;
        const uint64_t delay_us = (1000000ULL << shift);
        s_attempt++;
        ESP_LOGW(TAG, "disconnected (reason %d), retry in %llu s",
                 ((wifi_event_sta_disconnected_t *)data)->reason, delay_us / 1000000ULL);
        esp_timer_start_once(s_retry_timer, delay_us);
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *ev = (const ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "got ip " IPSTR, IP2STR(&ev->ip_info.ip));
        s_attempt = 0;
        xEventGroupSetBits(s_group, CONNECTED_BIT);
    }
}

esp_err_t wifi_start(void)
{
    if (strlen(CONFIG_EW_WIFI_SSID) == 0) {
        return ESP_ERR_INVALID_STATE;
    }
    s_group = xEventGroupCreateStatic(&s_group_mem);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    const wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL, NULL));

    const esp_timer_create_args_t targs = {.callback = retry_cb, .name = "wifi_retry"};
    ESP_ERROR_CHECK(esp_timer_create(&targs, &s_retry_timer));

    wifi_config_t wc = {0};
    strlcpy((char *)wc.sta.ssid, CONFIG_EW_WIFI_SSID, sizeof(wc.sta.ssid));
    strlcpy((char *)wc.sta.password, CONFIG_EW_WIFI_PASSWORD, sizeof(wc.sta.password));
    wc.sta.threshold.authmode = strlen(CONFIG_EW_WIFI_PASSWORD) ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "connecting to \"%s\"", CONFIG_EW_WIFI_SSID);
    return ESP_OK;
}

bool wifi_wait_connected(uint32_t timeout_ms)
{
    if (s_group == NULL) {
        return false;
    }
    return xEventGroupWaitBits(s_group, CONNECTED_BIT, pdFALSE, pdTRUE, pdMS_TO_TICKS(timeout_ms)) &
           CONNECTED_BIT;
}

bool wifi_is_connected(void)
{
    return s_group != NULL && (xEventGroupGetBits(s_group) & CONNECTED_BIT);
}

int wifi_rssi(void)
{
    wifi_ap_record_t ap;
    return (wifi_is_connected() && esp_wifi_sta_get_ap_info(&ap) == ESP_OK) ? ap.rssi : 0;
}
