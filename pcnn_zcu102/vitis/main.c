/*
 * PCNN accelerator -- bare-metal host application for ZCU102 (Cortex-A53).
 *
 * Requires the HLS-exported IP in the Vivado block design (AXI4-Lite on
 * HPM0-LPD/FPD) and the generated driver (xpcnn_step.h) in the Vitis BSP.
 *
 * Feature order (normalized to [0.1, 0.9]):
 *   x[0] outdoor T [C], x[1] solar [W/m2], x[2] occupancy count,
 *   x[3] room T [C],    x[4] HVAC load [W], x[5] case (-1 cool /0 off /1 heat)
 */
#include <stdio.h>
#include "xparameters.h"
#include "xpcnn_step.h"

#include "sleep.h"

#include "pcnn_config.h"     /* RAW_MIN/RAW_MAX + column defines             */
#include "demo_sequence.h"   /* optional: replayed test day (see README)     */

#define MODE_RESET_WARM 0
#define MODE_WARM       1
#define MODE_PRED       2
#define WARM_LEN        8

static XPcnn_step ip;

static float norm_feat(float raw, int col) {
    return 0.8f * (raw - RAW_MIN[col]) / (RAW_MAX[col] - RAW_MIN[col]) + 0.1f;
}
static float denorm_temp(float n) {
    return (n - 0.1f) / 0.8f * (RAW_MAX[COL_T] - RAW_MIN[COL_T]) + RAW_MIN[COL_T];
}

/* run one accelerator step; x = 6 normalized features */
static float pcnn_step_hw(const float x[6], int mode) {
    union { float f; u32 u; } cv;
    u32 buf[6];
    for (int i = 0; i < 6; i++) { cv.f = x[i]; buf[i] = cv.u; }
    XPcnn_step_Write_x_Words(&ip, 0, (word_type *)buf, 6);
    XPcnn_step_Set_mode(&ip, (u32)mode);
    XPcnn_step_Start(&ip);
    while (!XPcnn_step_IsDone(&ip)) { }
    cv.u = XPcnn_step_Get_T_out(&ip);
    return cv.f;
}

int main(void) {
    printf("\r\nPCNN on ZCU102 -- bare-metal demo\r\n");
    if (XPcnn_step_Initialize(&ip, XPAR_PCNN_STEP_0_BASEADDR) != XST_SUCCESS) {
        printf("IP init failed\r\n");
        return -1;
    }

#ifndef DEMO_HORIZONS
#define DEMO_HORIZONS 1
#endif
#ifndef DEMO_SEQ
#define DEMO_SEQ 96
#endif

    /* Replay test data stored in demo_sequence.h.
     * Supports multiple 96-step horizons for Harvard dataset evaluation. */
    double total_mse = 0.0;
    int    total_pred = 0;

    for (int h = 0; h < DEMO_HORIZONS; h++) {
        int base = h * 96;
        double horizon_mse = 0.0;
        int    horizon_pred = 0;

        for (int s = 0; s < 96; s++) {
            int t = base + s;
            int mode = (s == 0) ? MODE_RESET_WARM
                     : (s < WARM_LEN) ? MODE_WARM : MODE_PRED;
            float T = pcnn_step_hw(DEMO_X[t], mode);

            /* Only count prediction steps (after warm-start) for RMSE */
            if (s >= WARM_LEN) {
                double e = (double)T - (double)DEMO_REF[t];
                horizon_mse += e * e;
                horizon_pred++;
            }

            if (h < 3 && (s < 4 || s % 24 == 0))
                printf("h=%d t=%2d  T=%.3f C (gt %.3f C)\r\n",
                       h, s, denorm_temp(T), denorm_temp(DEMO_REF[t]));
        }

        double h_rmse = 0.0;
        if (horizon_pred > 0) {
            /* Newton's method sqrt (bare-metal, no libm double sqrt) */
            double val = horizon_mse / horizon_pred;
            double guess = val;
            for (int i = 0; i < 20 && guess > 0; i++)
                guess = 0.5 * (guess + val / guess);
            h_rmse = guess;
        }

        total_mse += horizon_mse;
        total_pred += horizon_pred;

        if (DEMO_HORIZONS <= 10 || h % (DEMO_HORIZONS / 10) == 0)
            printf("horizon %3d/%d  RMSE(norm)=%.6f  (~%.3f degC)\r\n",
                   h, DEMO_HORIZONS, h_rmse,
                   h_rmse * (RAW_MAX[COL_T] - RAW_MIN[COL_T]) / 0.8f);
    }

    /* Overall RMSE across all prediction steps */
    double overall_rmse = 0.0;
    if (total_pred > 0) {
        overall_rmse = total_mse / total_pred;
        for (int i = 0; i < 20; i++)
            overall_rmse = 0.5 * (overall_rmse + (total_mse / total_pred) / overall_rmse);
    }
    printf("\r\n=== Overall: %d horizons, %d prediction steps ===\r\n",
           DEMO_HORIZONS, total_pred);
    printf("RMSE (normalized): %.6f\r\n", overall_rmse);
    printf("RMSE (degC):       %.3f\r\n",
           overall_rmse * (RAW_MAX[COL_T] - RAW_MIN[COL_T]) / 0.8f);

    /* In deployment: at each 15-min tick read sensors, normalize with
     * norm_feat(), call pcnn_step_hw(). Re-warm-start with measured room
     * temperature at the start of every prediction horizon. */
    return 0;
}
