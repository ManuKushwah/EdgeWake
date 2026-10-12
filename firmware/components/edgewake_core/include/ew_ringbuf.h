/*
 * ew_ringbuf.h - lock-free single-producer / single-consumer ring buffer of
 * int16 PCM samples.
 *
 * Why it needs no mutex: `head` is written only by the producer and `tail`
 * only by the consumer ("one writer per index"). Each side merely *reads* the
 * other side's index, and C11 acquire/release ordering guarantees that the
 * sample data is visible before the index that publishes it.
 *
 * Indices are free-running uint32 counters (never masked when stored), so
 *   used = head - tail      (correct even across 2^32 wrap-around)
 * and "full" vs "empty" can be told apart without sacrificing a slot.
 * Capacity must be a power of two so `index & mask` replaces a modulo.
 *
 * Overflow policy: a write that does not fit is dropped *as a whole* and
 * counted in `dropped`. The producer is not allowed to advance `tail` (that
 * would make it a second writer of the consumer's index), so "drop newest"
 * is the only lock-free policy. Dropped samples surface in health telemetry.
 */
#ifndef EW_RINGBUF_H
#define EW_RINGBUF_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Keeps head and tail on separate cache lines on cached targets (host CPUs).
 * Harmless on the ESP32-S3, where internal SRAM is not behind a data cache. */
#define EW_RB_PAD 64

typedef struct {
    int16_t *buf;          /* caller-owned, statically allocated storage */
    uint32_t mask;         /* capacity - 1 */
    _Alignas(EW_RB_PAD) _Atomic uint32_t head;    /* producer-owned */
    _Atomic uint32_t dropped;                     /* producer-owned stat */
    _Alignas(EW_RB_PAD) _Atomic uint32_t tail;    /* consumer-owned */
} ew_ringbuf_t;

/* Returns false if capacity is not a power of two (or zero / too large). */
bool ew_ringbuf_init(ew_ringbuf_t *rb, int16_t *storage, uint32_t capacity);

/* Producer side. Copies all `n` samples or none; returns true on success. */
bool ew_ringbuf_write(ew_ringbuf_t *rb, const int16_t *src, uint32_t n);

/* Consumer side. Copies exactly `n` samples if available; else copies none. */
bool ew_ringbuf_read(ew_ringbuf_t *rb, int16_t *dst, uint32_t n);

/* Either side may call these; the answer can be stale by the time it returns
 * but is always conservative for the caller's own role. */
uint32_t ew_ringbuf_used(ew_ringbuf_t *rb);
uint32_t ew_ringbuf_capacity(ew_ringbuf_t *rb);
uint32_t ew_ringbuf_dropped(ew_ringbuf_t *rb);

#ifdef __cplusplus
}
#endif

#endif /* EW_RINGBUF_H */
