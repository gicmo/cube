#ifndef SCA_H
#define SCA_H

#include <stdint.h>

#include <cube_ica.h>
#include <cube_io.h>

#include <ica.h>

#ifdef __cplusplus
extern "C" {
#endif
  typedef struct _sca_config_t sca_config_t;

  int   sca_config_count (cube_h5_t sca);
  char *sca_config_name (cube_h5_t sca, int index);
  sca_config_t * sca_config_read (cube_h5_t sca, const char *id);
  void sca_config_free (sca_config_t *cfg);
  const char *sca_config_get_id (sca_config_t *cfg);
  ica_prior_t * sca_config_create_prior (cube_t *ctx, sca_config_t *cfg, int dimension);
  cube_array_t * sca_config_create_gradient (cube_t *ctx, sca_config_t *cfg);

  int   sca_dataset_count (cube_h5_t sca, const char *config);
  char *sca_dataset_name (cube_h5_t sca, const char *config, int index);
  ica_dataset_t *sca_dataset_read (cube_h5_t h5, const char *config, const char *dataset);

  int sca_write_model (cube_h5_t h5, ica_t *ica);
  int sca_write_log (cube_h5_t h5, ica_t *ica, int i);

  int sca_model_read (cube_h5_t h5, ica_model_t *model, const char *cfgid, const char *id);

#ifdef __cplusplus
}
#endif

#endif /* SCA_H */
