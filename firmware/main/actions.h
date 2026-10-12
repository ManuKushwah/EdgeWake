/*
 * actions.h - what a recognized keyword does locally (no network involved).
 *
 *   "on" / "off"  -> LED GPIO high / low
 *   "go" / "stop" -> auxiliary GPIO (e.g. relay) high / low
 *   any keyword   -> pulse on the ACTION probe pin (latency measurement)
 */
#ifndef ACTIONS_H
#define ACTIONS_H

void actions_init(void);

/* Called from the KWS task the moment a keyword fires. */
void actions_on_keyword(int class_index);

#endif /* ACTIONS_H */
