#ifndef MONITOR_STDOUT_H
#define MONITOR_STDOUT_H

#include <ica.h>

#ifdef __cplusplus
extern "C" {
#endif

  void monitor_stdout_new (ica_t *ica, int verbose, int update_freq);

#ifdef __cplusplus
}
#endif

#endif
