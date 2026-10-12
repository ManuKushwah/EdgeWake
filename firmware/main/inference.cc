/*
 * inference.cc - TensorFlow Lite Micro wrapper for the int8 DS-CNN.
 *
 * TFLM never calls malloc: every tensor lives in one caller-provided arena
 * whose layout is planned once in AllocateTensors(). The interpreter and op
 * resolver objects are constructed with placement new into static storage,
 * so the whole inference path is statically allocated. On the ESP32-S3 the
 * esp-tflite-micro component swaps in ESP-NN kernels that use the S3's SIMD
 * instructions for the int8 convolutions.
 */
#include "inference.h"

#include <cstring>
#include <new>

#include "esp_log.h"
#include "ew_config.h"
#include "model_data.h"
#include "probes.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

const char *TAG = "infer";

// Activations + ESP-NN scratch buffers + interpreter bookkeeping. 32 KB was
// ~2.6 KB short (found by booting the image in QEMU); the actual use is
// logged at boot so regressions are visible.
constexpr size_t kArenaSize = 40 * 1024;
alignas(16) uint8_t s_arena[kArenaSize];

// Exactly the kernels the converted model uses (training/results/metrics.json
// "ops"); the static batch of 1 lets the converter fold the flatten away.
using OpResolver = tflite::MicroMutableOpResolver<5>;
alignas(OpResolver) uint8_t s_resolver_mem[sizeof(OpResolver)];
alignas(tflite::MicroInterpreter) uint8_t s_interp_mem[sizeof(tflite::MicroInterpreter)];

tflite::MicroInterpreter *s_interp = nullptr;
TfLiteTensor *s_input = nullptr;
TfLiteTensor *s_output = nullptr;

}  // namespace

extern "C" esp_err_t inference_init(inference_info_t *info)
{
    const tflite::Model *model = tflite::GetModel(g_kws_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "model schema %lu != runtime %d", (unsigned long)model->version(),
                 TFLITE_SCHEMA_VERSION);
        return ESP_ERR_INVALID_VERSION;
    }

    auto *resolver = new (s_resolver_mem) OpResolver();
    resolver->AddConv2D();
    resolver->AddDepthwiseConv2D();
    resolver->AddAveragePool2D();
    resolver->AddFullyConnected();
    resolver->AddSoftmax();

    s_interp = new (s_interp_mem) tflite::MicroInterpreter(model, *resolver, s_arena, kArenaSize);
    if (s_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed (arena %u bytes too small?)", (unsigned)kArenaSize);
        return ESP_ERR_NO_MEM;
    }

    s_input = s_interp->input(0);
    s_output = s_interp->output(0);
    if (s_input->type != kTfLiteInt8 || s_input->bytes != EW_INPUT_SIZE ||
        s_output->type != kTfLiteInt8 || s_output->bytes != EW_NUM_CLASSES) {
        ESP_LOGE(TAG, "unexpected tensor shapes/types: in %u bytes, out %u bytes",
                 (unsigned)s_input->bytes, (unsigned)s_output->bytes);
        return ESP_ERR_INVALID_SIZE;
    }

    info->input_scale = s_input->params.scale;
    info->input_zero_point = s_input->params.zero_point;
    info->output_scale = s_output->params.scale;
    info->output_zero_point = s_output->params.zero_point;
    info->arena_used_bytes = s_interp->arena_used_bytes();
    info->model_bytes = g_kws_model_len;
    ESP_LOGI(TAG, "model %u B, arena used %u / %u B, in q=(%.5f,%ld) out q=(%.5f,%ld)",
             (unsigned)g_kws_model_len, (unsigned)info->arena_used_bytes, (unsigned)kArenaSize,
             (double)info->input_scale, (long)info->input_zero_point,
             (double)info->output_scale, (long)info->output_zero_point);
    return ESP_OK;
}

extern "C" int inference_run(void *ctx, const int8_t *input, int8_t *scores)
{
    (void)ctx;
    std::memcpy(s_input->data.int8, input, EW_INPUT_SIZE);
    EW_PROBE_HIGH(EW_PROBE_INFER);
    const TfLiteStatus status = s_interp->Invoke();
    EW_PROBE_LOW(EW_PROBE_INFER);
    if (status != kTfLiteOk) {
        return -1;
    }
    std::memcpy(scores, s_output->data.int8, EW_NUM_CLASSES);
    return 0;
}
