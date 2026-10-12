/*
 * probes.h - GPIO timing probes for a logic analyzer.
 *
 * Writing the GPIO W1TS / W1TC ("write 1 to set / clear") registers directly
 * takes a single store instruction, so the probe itself adds almost no
 * timing error, unlike gpio_set_level(), which goes through argument checks
 * and a HAL call. Pins 0-31 live in GPIO_OUT_*, pins 32-48 in GPIO_OUT1_*.
 *
 * Channels on the analyzer:
 *   AUDIO  - high while a 20 ms DMA chunk is converted and pushed to the ring
 *   DSP    - high while the KWS task processes one hop (VAD + MFCC [+ CNN])
 *   INFER  - high during TFLite Micro Invoke()
 *   ACTION - pulse when a keyword action fires
 */
#ifndef PROBES_H
#define PROBES_H

#include "sdkconfig.h"

#if CONFIG_EW_DEBUG_PROBES
#include "soc/gpio_reg.h"
#include "soc/soc.h"

#define EW_PROBE_SET_REG(pin)   (((pin) < 32) ? GPIO_OUT_W1TS_REG : GPIO_OUT1_W1TS_REG)
#define EW_PROBE_CLR_REG(pin)   (((pin) < 32) ? GPIO_OUT_W1TC_REG : GPIO_OUT1_W1TC_REG)
#define EW_PROBE_HIGH(pin)      REG_WRITE(EW_PROBE_SET_REG(pin), 1UL << ((pin) & 31))
#define EW_PROBE_LOW(pin)       REG_WRITE(EW_PROBE_CLR_REG(pin), 1UL << ((pin) & 31))

#define EW_PROBE_AUDIO   CONFIG_EW_PROBE_AUDIO_GPIO
#define EW_PROBE_DSP     CONFIG_EW_PROBE_DSP_GPIO
#define EW_PROBE_INFER   CONFIG_EW_PROBE_INFER_GPIO
#define EW_PROBE_ACTION  CONFIG_EW_PROBE_ACTION_GPIO
#else
#define EW_PROBE_HIGH(pin)      ((void)0)
#define EW_PROBE_LOW(pin)       ((void)0)
#endif

/* Configures the probe pins as outputs (no-op when probes are disabled). */
void probes_init(void);

#endif /* PROBES_H */
