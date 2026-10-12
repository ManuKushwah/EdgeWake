#include "actions.h"

#include "driver/gpio.h"
#include "model_settings.h"
#include "probes.h"
#include "sdkconfig.h"

void actions_init(void)
{
    const gpio_config_t cfg = {
        .pin_bit_mask = (1ULL << CONFIG_EW_LED_GPIO) | (1ULL << CONFIG_EW_AUX_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&cfg));
    gpio_set_level(CONFIG_EW_LED_GPIO, 0);
    gpio_set_level(CONFIG_EW_AUX_GPIO, 0);
}

void actions_on_keyword(int class_index)
{
    EW_PROBE_HIGH(EW_PROBE_ACTION);
    switch (class_index) {
    case EW_LABEL_ON:
        gpio_set_level(CONFIG_EW_LED_GPIO, 1);
        break;
    case EW_LABEL_OFF:
        gpio_set_level(CONFIG_EW_LED_GPIO, 0);
        break;
    case EW_LABEL_GO:
        gpio_set_level(CONFIG_EW_AUX_GPIO, 1);
        break;
    case EW_LABEL_STOP:
        gpio_set_level(CONFIG_EW_AUX_GPIO, 0);
        break;
    default:
        break;  /* other keywords are only reported via telemetry */
    }
    EW_PROBE_LOW(EW_PROBE_ACTION);
}
