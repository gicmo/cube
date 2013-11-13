
#ifndef CUBE_H
#define CUBE_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "cube_error.h"

typedef struct _cube_t cube_t;
typedef enum   _cube_status_t cube_status_t;

enum  _cube_memcpy_kind_t {

  CMK_HOST_2_HOST     = 0,
  CMK_HOST_2_DEVICE   = 1,
  CMK_DEVICE_2_HOST   = 2,
  CMK_DEVICE_2_DEVICE = 3,
  CMK_DEFAULT         = 4

};

typedef enum _cube_memcpy_kind_t cube_memcpy_kind_t;

cube_t * cube_context_new (int gpu);
void     cube_context_destroy (cube_t **ctx);

int      cube_context_check (cube_t *ctx);

int      cube_context_is_gpu (cube_t *ctx);

void *   cube_malloc_device (cube_t *ctx, size_t size);
void     cube_memset_device (cube_t *ctx, void *s, int c, size_t n);
void     cube_free_device (cube_t *ctx, void *dev_ptr);
void *   cube_host_register   (cube_t *ctx, void *host, size_t len);
int      cube_host_unregister (cube_t *ctx, void *host);

void *   cube_memcpy (cube_t *ctx,
		      void   *dest,
		      void   *src,
		      size_t  n,
		      cube_memcpy_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif /* CUBE_H */
