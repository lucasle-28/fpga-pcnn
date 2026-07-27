// PCNN single-zone accelerator for ZCU102 (XCZU9EG) -- Vitis HLS top header
// Model: EPFL PCNN (Di Natale et al.), config from Wang et al. 2026 Table 6:
//   input MLP 2->32 ReLU, LSTM 3x128, LayerNorm, output MLP 128->32->1 (tanh),
//   physics module E with parameters a,b,d (no neighbor zone -> c unused).
#ifndef PCNN_H
#define PCNN_H

#include "pcnn_config.h"

typedef float data_t;   // change to ap_fixed<W,I> for a quantized build

// mode values
#define PCNN_MODE_RESET_WARM 0  // reset internal state, then one warm-start step
#define PCNN_MODE_WARM       1  // warm-start step (true room temp fed in x[COL_T])
#define PCNN_MODE_PRED       2  // autoregressive prediction step

extern "C" void pcnn_step(const data_t x[N_FEAT], int mode,
                          data_t *T_out, data_t *D_out, data_t *E_out);

#endif
