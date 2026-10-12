#include <pthread.h>
#include <sched.h>
#include <stdint.h>

#include "ew_ringbuf.h"
#include "ew_test.h"

static void test_rejects_bad_capacity(void)
{
    static int16_t storage[100];
    ew_ringbuf_t rb;
    EW_CHECK(!ew_ringbuf_init(&rb, storage, 100));
    EW_CHECK(!ew_ringbuf_init(&rb, storage, 0));
    EW_CHECK(ew_ringbuf_init(&rb, storage, 64));
    EW_CHECK(ew_ringbuf_capacity(&rb) == 64);
}

static void test_write_read_roundtrip(void)
{
    static int16_t storage[16];
    ew_ringbuf_t rb;
    ew_ringbuf_init(&rb, storage, 16);

    int16_t in[5] = {1, 2, 3, 4, 5};
    int16_t out[5] = {0};
    EW_CHECK(!ew_ringbuf_read(&rb, out, 1));  /* empty */
    EW_CHECK(ew_ringbuf_write(&rb, in, 5));
    EW_CHECK(ew_ringbuf_used(&rb) == 5);
    EW_CHECK(!ew_ringbuf_read(&rb, out, 6));  /* not enough: copies nothing */
    EW_CHECK(ew_ringbuf_used(&rb) == 5);
    EW_CHECK(ew_ringbuf_read(&rb, out, 5));
    for (int i = 0; i < 5; i++) {
        EW_CHECK(out[i] == in[i]);
    }
    EW_CHECK(ew_ringbuf_used(&rb) == 0);
}

static void test_full_drops_newest(void)
{
    static int16_t storage[8];
    ew_ringbuf_t rb;
    ew_ringbuf_init(&rb, storage, 8);

    int16_t in[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    EW_CHECK(ew_ringbuf_write(&rb, in, 8));   /* exactly full: no wasted slot */
    EW_CHECK(ew_ringbuf_used(&rb) == 8);
    EW_CHECK(!ew_ringbuf_write(&rb, in, 3));  /* doesn't fit: whole chunk dropped */
    EW_CHECK(ew_ringbuf_dropped(&rb) == 3);

    int16_t out[8];
    EW_CHECK(ew_ringbuf_read(&rb, out, 8));   /* the old data is intact */
    for (int i = 0; i < 8; i++) {
        EW_CHECK(out[i] == i);
    }
}

static void test_wraparound(void)
{
    static int16_t storage[8];
    ew_ringbuf_t rb;
    ew_ringbuf_init(&rb, storage, 8);

    int16_t in[6], out[6];
    int16_t next_in = 0, next_out = 0;
    for (int round = 0; round < 50; round++) {
        for (int i = 0; i < 6; i++) {
            in[i] = next_in++;
        }
        EW_CHECK(ew_ringbuf_write(&rb, in, 6));
        EW_CHECK(ew_ringbuf_read(&rb, out, 6));
        for (int i = 0; i < 6; i++) {
            EW_CHECK(out[i] == next_out++);
        }
    }
}

static void test_index_overflow(void)
{
    /* Free-running indices wrap at 2^32; `head - tail` must stay correct. */
    static int16_t storage[8];
    ew_ringbuf_t rb;
    ew_ringbuf_init(&rb, storage, 8);
    atomic_store(&rb.head, UINT32_MAX - 2);
    atomic_store(&rb.tail, UINT32_MAX - 2);

    int16_t in[6] = {10, 11, 12, 13, 14, 15}, out[6] = {0};
    EW_CHECK(ew_ringbuf_used(&rb) == 0);
    EW_CHECK(ew_ringbuf_write(&rb, in, 6));   /* head wraps past zero here */
    EW_CHECK(ew_ringbuf_used(&rb) == 6);
    EW_CHECK(ew_ringbuf_read(&rb, out, 6));
    for (int i = 0; i < 6; i++) {
        EW_CHECK(out[i] == in[i]);
    }
}

/* --- concurrent stress: one producer thread, one consumer thread ---------- */

#define STRESS_SAMPLES 4000000u
#define STRESS_CAPACITY 1024u

static int16_t stress_storage[STRESS_CAPACITY];
static ew_ringbuf_t stress_rb;
static int stress_errors;

static void *producer(void *arg)
{
    (void)arg;
    int16_t chunk[96];
    uint32_t sent = 0;
    unsigned seed = 1;
    while (sent < STRESS_SAMPLES) {
        seed = seed * 1103515245u + 12345u;
        uint32_t n = 1 + (seed >> 16) % 96;
        if (n > STRESS_SAMPLES - sent) {
            n = STRESS_SAMPLES - sent;
        }
        for (uint32_t i = 0; i < n; i++) {
            chunk[i] = (int16_t)(sent + i);
        }
        while (!ew_ringbuf_write(&stress_rb, chunk, n)) {
            sched_yield();  /* full: real firmware drops, the test retries */
        }
        sent += n;
    }
    return NULL;
}

static void *consumer(void *arg)
{
    (void)arg;
    int16_t chunk[80];
    uint32_t got = 0;
    unsigned seed = 7;
    while (got < STRESS_SAMPLES) {
        seed = seed * 1103515245u + 12345u;
        uint32_t n = 1 + (seed >> 16) % 80;
        if (n > STRESS_SAMPLES - got) {
            n = STRESS_SAMPLES - got;
        }
        if (!ew_ringbuf_read(&stress_rb, chunk, n)) {
            sched_yield();
            continue;
        }
        for (uint32_t i = 0; i < n; i++) {
            if (chunk[i] != (int16_t)(got + i)) {
                stress_errors++;
            }
        }
        got += n;
    }
    return NULL;
}

static void test_concurrent_spsc(void)
{
    ew_ringbuf_init(&stress_rb, stress_storage, STRESS_CAPACITY);
    stress_errors = 0;
    pthread_t p, c;
    pthread_create(&c, NULL, consumer, NULL);
    pthread_create(&p, NULL, producer, NULL);
    pthread_join(p, NULL);
    pthread_join(c, NULL);
    EW_CHECK(stress_errors == 0);
    EW_CHECK(ew_ringbuf_used(&stress_rb) == 0);
}

int main(void)
{
    printf("ringbuf\n");
    EW_RUN(test_rejects_bad_capacity);
    EW_RUN(test_write_read_roundtrip);
    EW_RUN(test_full_drops_newest);
    EW_RUN(test_wraparound);
    EW_RUN(test_index_overflow);
    EW_RUN(test_concurrent_spsc);
    return EW_TEST_RESULT();
}
