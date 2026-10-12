#include "probes.h"

#include "driver/gpio.h"

void probes_init(void)
{
#if CONFIG_EW_DEBUG_PROBES
    const gpio_config_t cfg = {
        .pin_bit_mask = (1ULL << EW_PROBE_AUDIO) | (1ULL << EW_PROBE_DSP) |
                        (1ULL << EW_PROBE_INFER) | (1ULL << EW_PROBE_ACTION),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&cfg));
    EW_PROBE_LOW(EW_PROBE_AUDIO);
    EW_PROBE_LOW(EW_PROBE_DSP);
    EW_PROBE_LOW(EW_PROBE_INFER);
    EW_PROBE_LOW(EW_PROBE_ACTION);
#endif
}
