#include "ew_vad.h"

#include <math.h>

#include "ew_port.h"

void ew_vad_default_config(ew_vad_config_t *cfg)
{
    cfg->k_sigma = 3.0f;
    cfg->min_margin_db = 6.0f;     /* at least 4x the background energy */
    cfg->abs_min_db = 30.0f;       /* E = 1000 LSB^2, ~ -60 dBFS RMS */
    cfg->noise_alpha = 0.05f;      /* ~20 hops = 0.4 s memory */
    cfg->drift_alpha = 0.002f;     /* a +12 dB step is absorbed in ~7 s */
    cfg->hangover_hops = 15;       /* 300 ms */
    cfg->warmup_hops = 25;         /* 500 ms */
}

void ew_vad_init(ew_vad_t *v, const ew_vad_config_t *cfg)
{
    v->cfg = *cfg;
    v->mu_db = 0.0f;
    v->var_db = 1.0f;
    v->last_db = 0.0f;
    v->hangover_left = 0;
    v->hops_seen = 0;
    v->seeded = false;
    v->active = true;
}

EW_IRAM float ew_vad_energy(const int16_t *x, size_t n)
{
    if (n == 0) {
        return 0.0f;
    }
    /* int16^2 < 2^30, so an int64 sum is exact for any realistic n. */
    int64_t acc = 0;
    for (size_t i = 0; i < n; i++) {
        acc += (int32_t)x[i] * (int32_t)x[i];
    }
    return (float)acc / (float)n;
}

EW_IRAM bool ew_vad_process(ew_vad_t *v, const int16_t *x, size_t n)
{
    const float level = 10.0f * log10f(ew_vad_energy(x, n) + 1.0f);
    v->last_db = level;
    if (!v->seeded) {
        v->mu_db = level;
        v->seeded = true;
    }

    const float spread = v->cfg.k_sigma * sqrtf(v->var_db);
    const float threshold = v->mu_db + (spread > v->cfg.min_margin_db ? spread : v->cfg.min_margin_db);
    const bool speech = (level > threshold) && (level > v->cfg.abs_min_db);

    if (!speech && v->hangover_left == 0) {
        /* Gate closed: this hop is background, learn from it. */
        const float d = level - v->mu_db;
        v->mu_db += v->cfg.noise_alpha * d;
        v->var_db += v->cfg.noise_alpha * (d * d - v->var_db);
    } else {
        /* Gate open: only drift, so the gate cannot lock open forever. */
        v->mu_db += v->cfg.drift_alpha * (level - v->mu_db);
    }

    if (speech) {
        v->hangover_left = v->cfg.hangover_hops;
        v->active = true;
    } else if (v->hangover_left > 0) {
        v->hangover_left--;
        v->active = true;
    } else {
        v->active = false;
    }

    if (v->hops_seen < v->cfg.warmup_hops) {
        v->hops_seen++;
        v->active = true;
    }
    return v->active;
}
