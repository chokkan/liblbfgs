#ifndef __LIBLBFGS_GO_WRAP_H__
#define __LIBLBFGS_GO_WRAP_H__

#include <lbfgs.h>

// Exported function from lbfgs.go.
extern lbfgsfloatval_t goLbfgsEvaluate(void *instance, lbfgsfloatval_t *x,
                                       lbfgsfloatval_t *g, int n,
                                       lbfgsfloatval_t step);

// Exported function from lbfgs.go.
extern int goLbfgsProgress(void *instance, lbfgsfloatval_t *x,
                           lbfgsfloatval_t *g, lbfgsfloatval_t fx,
                           lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
                           lbfgsfloatval_t step, int n, int k, int ls);

// Uses evaluate and progress function exported from go to call C lbfgs.
static inline int goLbfgs(int n, lbfgsfloatval_t *x, lbfgsfloatval_t *ptr_fx,
                          void *instance, lbfgs_parameter_t *param) {
  return lbfgs(n, x, ptr_fx, (lbfgs_evaluate_t)goLbfgsEvaluate,
               (lbfgs_progress_t)goLbfgsProgress, instance, param);
}

#endif  // __LIBLBFGS_GO_WRAP_H__
