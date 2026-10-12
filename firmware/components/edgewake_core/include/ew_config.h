/*
 * ew_config.h - single source of truth for the audio front-end geometry.
 *
 * Every number here is baked into the trained model: the training pipeline
 * computes its features by calling this same C code (see training/frontend.py),
 * so changing anything below means re-training and re-exporting the model.
 */
#ifndef EW_CONFIG_H
#define EW_CONFIG_H

#define EW_SAMPLE_RATE_HZ   16000

/* 32 ms analysis window (= FFT length, so no zero padding), 20 ms hop. */
#define EW_FRAME_LEN        512
#define EW_FRAME_HOP        320
#define EW_FFT_LEN          512
#define EW_FFT_BINS         (EW_FFT_LEN / 2 + 1)   /* 257 one-sided bins */

/* Mel filterbank + DCT. 40 bands over 20 Hz - 4 kHz, keep 10 cepstra. */
#define EW_MEL_BANDS        40
#define EW_MEL_LOW_HZ       20.0f
#define EW_MEL_HIGH_HZ      4000.0f
#define EW_NUM_MFCC         10
#define EW_LOG_EPS          1e-6f

/* One model input = 1 s of audio = (16000 - 512) / 320 + 1 = 49 frames. */
#define EW_NUM_FRAMES       49
#define EW_INPUT_SIZE       (EW_NUM_FRAMES * EW_NUM_MFCC)  /* 490 int8 values */

/* Speech Commands 12-class task: silence, unknown + 10 keywords. */
#define EW_NUM_CLASSES      12

#endif /* EW_CONFIG_H */
