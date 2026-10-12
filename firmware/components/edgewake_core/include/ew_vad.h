/*
 * ew_vad.h - energy-based voice activity detector used as an inference gate.
 *
 * For each 20 ms hop:
 *   E = mean(x^2)  (exact int64 accumulation of int16 samples)
 *   L = 10*log10(E + 1)                       level in dB
 *   speech = L > mu + max(k * sigma, margin)  AND  L > abs_min
 *
 * mu / sigma are a running mean and standard deviation of the background
 * level in dB. Working in dB and scaling the threshold with sigma means the
 * gate adapts to both how loud the background is and how much it fluctuates:
 * steady fan noise gets a threshold close to its level, bursty or
 * low-frequency-heavy noise gets a higher one. (A fixed "4x the minimum
 * energy" rule opened on ~70 % of hops of pink noise for that reason.)
 *
 * The statistics are updated only while the gate is fully closed, so speech
 * and its hangover tail do not inflate them. While the gate is open, mu only
 * drifts slowly toward the input; a lasting level change (a fan switching
 * on) is therefore absorbed within seconds and the gate cannot stick open.
 *
 * Exactly-zero input (a MEMS mic outputs zeros while it starts up) carries
 * no information about the room and is ignored by the
 * statistics; seeding the background from it would latch the gate open for
 * tens of seconds once real noise arrives. During warm-up the statistics are
 * updated on every hop with a fast rate so they lock onto the room quickly.
 *
 * Hangover keeps the gate open after the last speech hop: the model looks at
 * a 1 s window, so the keyword is still inside that window after the speaker
 * stops, and trailing low-energy phonemes ("-p" in "stop") must not be cut.
 */
#ifndef EW_VAD_H
#define EW_VAD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float k_sigma;            /* threshold = mu + max(k_sigma * sigma, min_margin_db) */
    float min_margin_db;      /* never closer than this to the background */
    float abs_min_db;         /* ignore anything quieter (dB re 1 LSB^2) */
    float noise_alpha;        /* background statistics update rate (gate closed) */
    float warmup_alpha;       /* faster update rate used during warm-up */
    float drift_alpha;        /* slow drift of mu while the gate is open */
    uint32_t hangover_hops;   /* keep gate open this many hops after speech */
    uint32_t warmup_hops;     /* gate forced open while the statistics settle */
} ew_vad_config_t;

typedef struct {
    ew_vad_config_t cfg;
    float mu_db;              /* background level */
    float var_db;             /* background variance (dB^2) */
    float last_db;
    uint32_t hangover_left;
    uint32_t hops_seen;
    bool seeded;
    bool active;
} ew_vad_t;

/* Defaults for a 16 kHz, 20 ms hop front-end (tuned on host stream replays,
 * see training/stream_eval.py). */
void ew_vad_default_config(ew_vad_config_t *cfg);

void ew_vad_init(ew_vad_t *v, const ew_vad_config_t *cfg);

/* Feeds one hop of samples; returns true while the gate is open. */
bool ew_vad_process(ew_vad_t *v, const int16_t *x, size_t n);

/* Mean-square energy of a block, exposed for tests. */
float ew_vad_energy(const int16_t *x, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* EW_VAD_H */
