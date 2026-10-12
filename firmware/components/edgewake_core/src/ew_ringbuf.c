#include "ew_ringbuf.h"

#include <string.h>

#include "ew_port.h"

bool ew_ringbuf_init(ew_ringbuf_t *rb, int16_t *storage, uint32_t capacity)
{
    if (rb == NULL || storage == NULL || capacity == 0 ||
        (capacity & (capacity - 1)) != 0 || capacity > (1u << 30)) {
        return false;
    }
    rb->buf = storage;
    rb->mask = capacity - 1;
    atomic_init(&rb->head, 0);
    atomic_init(&rb->tail, 0);
    atomic_init(&rb->dropped, 0);
    return true;
}

EW_IRAM bool ew_ringbuf_write(ew_ringbuf_t *rb, const int16_t *src, uint32_t n)
{
    /* Our own index: nobody else writes it, relaxed is enough. */
    const uint32_t head = atomic_load_explicit(&rb->head, memory_order_relaxed);
    /* The consumer's index: acquire, so its reads of those slots are finished
     * before we overwrite them. */
    const uint32_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire);

    const uint32_t capacity = rb->mask + 1;
    if (EW_UNLIKELY(n > capacity - (head - tail))) {
        atomic_fetch_add_explicit(&rb->dropped, n, memory_order_relaxed);
        return false;
    }

    /* Copy in at most two segments: up to the end of storage, then wrap. */
    const uint32_t start = head & rb->mask;
    const uint32_t first = (n < capacity - start) ? n : capacity - start;
    memcpy(&rb->buf[start], src, first * sizeof(int16_t));
    memcpy(&rb->buf[0], src + first, (n - first) * sizeof(int16_t));

    /* Release: the samples above become visible before the new head does. */
    atomic_store_explicit(&rb->head, head + n, memory_order_release);
    return true;
}

EW_IRAM bool ew_ringbuf_read(ew_ringbuf_t *rb, int16_t *dst, uint32_t n)
{
    const uint32_t tail = atomic_load_explicit(&rb->tail, memory_order_relaxed);
    /* Acquire pairs with the producer's release: if we see the new head we
     * are guaranteed to see the samples written before it. */
    const uint32_t head = atomic_load_explicit(&rb->head, memory_order_acquire);

    if (head - tail < n) {
        return false;
    }

    const uint32_t capacity = rb->mask + 1;
    const uint32_t start = tail & rb->mask;
    const uint32_t first = (n < capacity - start) ? n : capacity - start;
    memcpy(dst, &rb->buf[start], first * sizeof(int16_t));
    memcpy(dst + first, &rb->buf[0], (n - first) * sizeof(int16_t));

    /* Release: our reads complete before the producer may reuse the slots. */
    atomic_store_explicit(&rb->tail, tail + n, memory_order_release);
    return true;
}

uint32_t ew_ringbuf_used(ew_ringbuf_t *rb)
{
    const uint32_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire);
    const uint32_t head = atomic_load_explicit(&rb->head, memory_order_acquire);
    return head - tail;
}

uint32_t ew_ringbuf_capacity(ew_ringbuf_t *rb)
{
    return rb->mask + 1;
}

uint32_t ew_ringbuf_dropped(ew_ringbuf_t *rb)
{
    return atomic_load_explicit(&rb->dropped, memory_order_relaxed);
}
