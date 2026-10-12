/*
 * inference.h - C interface to the TensorFlow Lite Micro (C++) runtime.
 */
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
    size_t arena_used_bytes;
    size_t model_bytes;
} inference_info_t;

/* Parses the model, registers kernels and plans the tensor arena.
 * No heap is used: interpreter, resolver and arena are all static. */
esp_err_t inference_init(inference_info_t *info);

/* input: EW_INPUT_SIZE int8 values; scores: EW_NUM_CLASSES int8 values.
 * Returns 0 on success. Signature matches ew_infer_fn. */
int inference_run(void *ctx, const int8_t *input, int8_t *scores);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
