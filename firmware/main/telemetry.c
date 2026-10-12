#include "telemetry.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include "app_config.h"
#include "cJSON.h"
#include "esp_log.h"
#include "health.h"
#include "kws_task.h"
#include "model_settings.h"
#include "mqtt_client.h"
#include "ota.h"
#include "sdkconfig.h"

static const char *TAG = "telemetry";

#if EW_HAVE_CERTS
/* Embedded by EMBED_TXTFILES (NUL-terminated). */
extern const char aws_root_ca[] asm("_binary_AmazonRootCA1_pem_start");
extern const char device_cert[] asm("_binary_device_pem_crt_start");
extern const char device_key[] asm("_binary_private_pem_key_start");

static esp_mqtt_client_handle_t s_client;
static QueueHandle_t s_events;
static atomic_bool s_connected;
static char s_uri[160];
static char s_topic_det[96], s_topic_health[96], s_topic_status[96], s_topic_cmd[96];
static char s_json[768];  /* used only by the telemetry task */

static StaticTask_t s_tcb;
static StackType_t s_stack[EW_STACK_TELEMETRY];

static void publish_status(const char *json)
{
    esp_mqtt_client_enqueue(s_client, s_topic_status, json, 0, 1, 0, true);
}

static void handle_command(const char *data, int len)
{
    cJSON *root = cJSON_ParseWithLength(data, (size_t)len);
    if (root == NULL) {
        ESP_LOGW(TAG, "ignoring malformed command");
        return;
    }
    const cJSON *url = cJSON_GetObjectItem(root, "ota_url");
    const cJSON *vad = cJSON_GetObjectItem(root, "vad");
    const cJSON *thr = cJSON_GetObjectItem(root, "threshold");
    if (cJSON_IsString(url)) {
        const esp_err_t err = ota_start_update(url->valuestring);
        publish_status(err == ESP_OK ? "{\"ota\":\"started\"}" : "{\"ota\":\"rejected\"}");
    }
    if (cJSON_IsBool(vad)) {
        kws_request_vad(cJSON_IsTrue(vad));
    }
    if (cJSON_IsNumber(thr) && thr->valuedouble > 0.0 && thr->valuedouble < 1.0) {
        kws_request_threshold((float)thr->valuedouble);
    }
    cJSON_Delete(root);
}

static void on_mqtt_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;
    (void)base;
    esp_mqtt_event_handle_t ev = (esp_mqtt_event_handle_t)data;
    switch ((esp_mqtt_event_id_t)id) {
    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "connected to AWS IoT Core");
        atomic_store(&s_connected, true);
        esp_mqtt_client_subscribe(s_client, s_topic_cmd, 1);
        publish_status("{\"online\":true}");
        break;
    case MQTT_EVENT_DISCONNECTED:
        ESP_LOGW(TAG, "disconnected (client reconnects automatically)");
        atomic_store(&s_connected, false);
        break;
    case MQTT_EVENT_DATA:
        /* Commands are small; ignore anything fragmented across events. */
        if (ev->topic_len == (int)strlen(s_topic_cmd) &&
            strncmp(ev->topic, s_topic_cmd, ev->topic_len) == 0 &&
            ev->current_data_offset == 0 && ev->data_len == ev->total_data_len) {
            handle_command(ev->data, ev->data_len);
        }
        break;
    case MQTT_EVENT_ERROR:
        ESP_LOGE(TAG, "mqtt error");
        break;
    default:
        break;
    }
}

static void telemetry_task(void *arg)
{
    (void)arg;
    const TickType_t period = pdMS_TO_TICKS(CONFIG_EW_HEALTH_PERIOD_S * 1000);
    TickType_t next_health = xTaskGetTickCount() + period;

    for (;;) {
        const TickType_t now = xTaskGetTickCount();
        const TickType_t wait = (next_health > now) ? next_health - now : 0;
        kws_event_t ev;
        if (xQueueReceive(s_events, &ev, wait) == pdTRUE && atomic_load(&s_connected)) {
            const int n = snprintf(s_json, sizeof(s_json),
                "{\"seq\":%lu,\"kw\":\"%s\",\"score\":%.3f,\"t_ms\":%lu,\"hop_to_action_us\":%lu}",
                (unsigned long)ev.seq, ew_labels[ev.class_index], (double)ev.score,
                (unsigned long)ev.timestamp_ms, (unsigned long)ev.hop_to_action_us);
            /* enqueue: copied into the outbox and sent by the MQTT task, so a
             * slow network never blocks this task either. */
            esp_mqtt_client_enqueue(s_client, s_topic_det, s_json, n, 1, 0, true);
        }
        if ((int32_t)(xTaskGetTickCount() - next_health) >= 0) {
            next_health += period;
            if (atomic_load(&s_connected)) {
                health_t h;
                health_snapshot(&h);
                const int n = health_to_json(&h, s_json, sizeof(s_json));
                if (n > 0 && n < (int)sizeof(s_json)) {
                    esp_mqtt_client_enqueue(s_client, s_topic_health, s_json, n, 0, 0, true);
                }
            }
        }
    }
}

esp_err_t telemetry_start(QueueHandle_t detection_events)
{
    if (strlen(CONFIG_EW_AWS_IOT_ENDPOINT) == 0) {
        return ESP_ERR_NOT_SUPPORTED;
    }
    s_events = detection_events;
    snprintf(s_uri, sizeof(s_uri), "mqtts://%s:8883", CONFIG_EW_AWS_IOT_ENDPOINT);
    snprintf(s_topic_det, sizeof(s_topic_det), "edgewake/%s/detections", CONFIG_EW_THING_NAME);
    snprintf(s_topic_health, sizeof(s_topic_health), "edgewake/%s/health", CONFIG_EW_THING_NAME);
    snprintf(s_topic_status, sizeof(s_topic_status), "edgewake/%s/status", CONFIG_EW_THING_NAME);
    snprintf(s_topic_cmd, sizeof(s_topic_cmd), "edgewake/%s/cmd", CONFIG_EW_THING_NAME);

    const esp_mqtt_client_config_t cfg = {
        .broker.address.uri = s_uri,
        .broker.verification.certificate = aws_root_ca,           /* server auth */
        .credentials.client_id = CONFIG_EW_THING_NAME,
        .credentials.authentication.certificate = device_cert,    /* client auth (mTLS) */
        .credentials.authentication.key = device_key,
        .session.keepalive = 60,
        .session.last_will = {
            .topic = s_topic_status,
            .msg = "{\"online\":false}",
            .qos = 1,
        },
        .network.reconnect_timeout_ms = 5000,
    };
    s_client = esp_mqtt_client_init(&cfg);
    if (s_client == NULL) {
        return ESP_FAIL;
    }
    ESP_ERROR_CHECK(esp_mqtt_client_register_event(s_client, ESP_EVENT_ANY_ID, on_mqtt_event, NULL));
    ESP_ERROR_CHECK(esp_mqtt_client_start(s_client));

    TaskHandle_t h = xTaskCreateStaticPinnedToCore(telemetry_task, "telemetry", EW_STACK_TELEMETRY,
                                                   NULL, EW_PRIO_TELEMETRY, s_stack, &s_tcb,
                                                   EW_CORE_NET);
    return h ? ESP_OK : ESP_FAIL;
}

bool telemetry_connected(void)
{
    return atomic_load(&s_connected);
}

#else /* !EW_HAVE_CERTS */

esp_err_t telemetry_start(QueueHandle_t detection_events)
{
    (void)detection_events;
    ESP_LOGW(TAG, "built without AWS IoT certificates: telemetry disabled");
    return ESP_ERR_NOT_SUPPORTED;
}

bool telemetry_connected(void)
{
    return false;
}

#endif
