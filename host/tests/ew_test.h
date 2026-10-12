/* ew_test.h - minimal assertion helpers for the host unit tests. */
#ifndef EW_TEST_H
#define EW_TEST_H

#include <math.h>
#include <stdio.h>

static int ew_test_failures;

#define EW_CHECK(cond)                                                          \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,    \
                    #cond);                                                     \
            ew_test_failures++;                                                 \
        }                                                                       \
    } while (0)

#define EW_CHECK_NEAR(a, b, tol)                                                \
    do {                                                                        \
        const double ew_a_ = (double)(a), ew_b_ = (double)(b);                  \
        if (!(fabs(ew_a_ - ew_b_) <= (double)(tol))) {                          \
            fprintf(stderr, "%s:%d: |%s - %s| = |%g - %g| > %g\n", __FILE__,    \
                    __LINE__, #a, #b, ew_a_, ew_b_, (double)(tol));             \
            ew_test_failures++;                                                 \
        }                                                                       \
    } while (0)

#define EW_RUN(test_fn)                                                         \
    do {                                                                        \
        const int before_ = ew_test_failures;                                   \
        test_fn();                                                              \
        printf("  %-40s %s\n", #test_fn,                                        \
               ew_test_failures == before_ ? "ok" : "FAILED");                  \
    } while (0)

#define EW_TEST_RESULT() (ew_test_failures == 0 ? 0 : 1)

/* Small deterministic PRNG (xorshift32) so tests are reproducible. */
static unsigned ew_rand_state = 0x12345678u;
static inline unsigned ew_rand(void)
{
    unsigned x = ew_rand_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return ew_rand_state = x;
}

#endif /* EW_TEST_H */
