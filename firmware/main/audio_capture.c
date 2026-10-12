/*
 * audio_capture.c - INMP441 -> I2S (DMA) -> int16 PCM -> lock-free ring.
 *
 * The I2S peripheral clocks the microphone at 16 kHz and its DMA engine
 * writes samples into a ring of 4 descriptors (20 ms each) without any CPU
 * involvement. i2s_channel_read() blocks this task until one descriptor is
 * complete, so capture costs CPU time only once per 20 ms, to convert the
 * samples and push them into the SPSC ring that the KWS task on core 1 reads.
 */
#include "audio_capture.h"

#include <stdatomic.h>
#include <string.h>

#include "app_config.h"
#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "ew_config.h"
#include "probes.h"
#include "sdkconfig.h"

static const char *TAG = "audio";

/* Everything on the audio path is statically allocated. */
static int16_t s_ring_storage[EW_AUDIO_RING_SAMPLES];
static ew_ringbuf_t s_ring;
static int32_t s_raw[EW_FRAME_HOP];     /* one DMA chunk of 32-bit I2S slots */
static int16_t s_pcm[EW_FRAME_HOP];

static StaticTask_t s_task_tcb;
static StackType_t s_task_stack[EW_STACK_AUDIO];
static TaskHandle_t s_consumer;
static TaskHandle_t s_task;
static i2s_chan_handle_t s_rx;

/* Written only by the capture task, read by others: relaxed atomics. */
static _Atomic uint32_t s_chunks;
static _Atomic uint32_t s_read_errors;
static _Atomic int32_t s_last_peak;
static _Atomic uint32_t s_last_active_ms;

void audio_ring_init(void)
{
    ew_ringbuf_init(&s_ring, s_ring_storage, EW_AUDIO_RING_SAMPLES);
}

ew_ringbuf_t *audio_ring(void)
{
    return &s_ring;
}

static esp_err_t i2s_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = EW_I2S_DMA_DESC_NUM;
    chan_cfg.dma_frame_num = EW_FRAME_HOP;  /* one descriptor = one 20 ms hop */
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, NULL, &s_rx), TAG, "i2s_new_channel");

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(EW_SAMPLE_RATE_HZ),
        /* INMP441: Philips I2S, 24-bit data left-justified in 32-bit slots. */
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = CONFIG_EW_I2S_BCLK_GPIO,
            .ws = CONFIG_EW_I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = CONFIG_EW_I2S_DIN_GPIO,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;  /* L/R pin tied to GND */
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_rx, &std_cfg), TAG, "init_std_mode");
    return i2s_channel_enable(s_rx);
}

static inline int16_t sat16(int32_t v)
{
    return (v > INT16_MAX) ? INT16_MAX : (v < INT16_MIN) ? INT16_MIN : (int16_t)v;
}

static void capture_task(void *arg)
{
    (void)arg;
    /* Created on core 0, so the I2S interrupt is allocated on core 0 too. */
    ESP_ERROR_CHECK(i2s_init());
    ESP_ERROR_CHECK(esp_task_wdt_add(NULL));
    ESP_LOGI(TAG, "I2S @ %d Hz, %d x %d-sample DMA buffers, ring %d samples",
             EW_SAMPLE_RATE_HZ, EW_I2S_DMA_DESC_NUM, EW_FRAME_HOP, EW_AUDIO_RING_SAMPLES);

    /* One-pole DC blocker, y[n] = x[n] - x[n-1] + 0.995 * y[n-1] (Q15),
     * cut-off ~13 Hz: removes the MEMS mic's DC offset, which would otherwise
     * bias the VAD energy and the first MFCC. Runs at 24-bit resolution. */
    int32_t x1 = 0, y1 = 0;
    const int gain_shift = CONFIG_EW_MIC_SHIFT - 8;

    for (;;) {
        size_t bytes = 0;
        const esp_err_t err = i2s_channel_read(s_rx, s_raw, sizeof(s_raw), &bytes, pdMS_TO_TICKS(100));
        esp_task_wdt_reset();
        if (err != ESP_OK || bytes != sizeof(s_raw)) {
            atomic_fetch_add_explicit(&s_read_errors, 1, memory_order_relaxed);
            continue;
        }

        EW_PROBE_HIGH(EW_PROBE_AUDIO);
        int32_t lo = INT32_MAX, hi = INT32_MIN, peak = 0;
        for (int i = 0; i < EW_FRAME_HOP; i++) {
            const int32_t x = s_raw[i] >> 8;  /* 24-bit signed sample */
            const int32_t y = x - x1 + (int32_t)(((int64_t)y1 * 32604) >> 15);
            x1 = x;
            y1 = y;
            const int16_t s = sat16(y >> gain_shift);
            s_pcm[i] = s;
            lo = (x < lo) ? x : lo;
            hi = (x > hi) ? x : hi;
            const int32_t a = (s < 0) ? -(int32_t)s : s;
            peak = (a > peak) ? a : peak;
        }
        /* Drop-newest on overflow; the ring counts what it dropped. */
        ew_ringbuf_write(&s_ring, s_pcm, EW_FRAME_HOP);
        EW_PROBE_LOW(EW_PROBE_AUDIO);

        xTaskNotifyGive(s_consumer);

        atomic_fetch_add_explicit(&s_chunks, 1, memory_order_relaxed);
        atomic_store_explicit(&s_last_peak, peak, memory_order_relaxed);
        if (hi - lo > 16) {  /* a floating / dead data line reads constant */
            atomic_store_explicit(&s_last_active_ms, (uint32_t)(esp_timer_get_time() / 1000),
                                  memory_order_relaxed);
        }
    }
}

esp_err_t audio_capture_start(TaskHandle_t consumer)
{
    s_consumer = consumer;
    s_task = xTaskCreateStaticPinnedToCore(capture_task, "audio", EW_STACK_AUDIO, NULL,
                                           EW_PRIO_AUDIO, s_task_stack, &s_task_tcb, EW_CORE_AUDIO);
    return s_task ? ESP_OK : ESP_FAIL;
}

TaskHandle_t audio_task_handle(void)
{
    return s_task;
}

void audio_get_stats(audio_stats_t *out)
{
    out->chunks = atomic_load_explicit(&s_chunks, memory_order_relaxed);
    out->read_errors = atomic_load_explicit(&s_read_errors, memory_order_relaxed);
    out->dropped_samples = ew_ringbuf_dropped(&s_ring);
    out->last_peak = atomic_load_explicit(&s_last_peak, memory_order_relaxed);
}

bool audio_capture_is_alive(void)
{
    const uint32_t last = atomic_load_explicit(&s_last_active_ms, memory_order_relaxed);
    const uint32_t now = (uint32_t)(esp_timer_get_time() / 1000);
    return last != 0 && (now - last) < 1000;
}
