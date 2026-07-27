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
#include "xtime_l.h"
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
    if (XPcnn_step_Initialize(&ip, XPAR_PCNN_STEP_0_DEVICE_ID) != XST_SUCCESS) {
        printf("IP init failed\r\n");
        return -1;
    }

    /* Replay one test day stored in demo_sequence.h:
     * DEMO_X[96][6] normalized inputs, DEMO_REF[96] golden T (normalized). */
    XTime t0, t1;
    double max_err = 0.0;
    XTime_GetTime(&t0);
    for (int t = 0; t < 96; t++) {
        int mode = (t == 0) ? MODE_RESET_WARM : (t < WARM_LEN) ? MODE_WARM : MODE_PRED;
        float T = pcnn_step_hw(DEMO_X[t], mode);
        double e = T - DEMO_REF[t]; if (e < 0) e = -e;
        if (e > max_err) max_err = e;
        if (t % 12 == 0)
            printf("t=%2d  T=%.3f C (ref %.3f C)\r\n",
                   t, denorm_temp(T), denorm_temp(DEMO_REF[t]));
    }
    XTime_GetTime(&t1);
    double us = 1.0 * (t1 - t0) / (COUNTS_PER_SECOND / 1000000.0);
    printf("96 steps in %.1f us  (%.1f us/step)\r\n", us, us / 96.0);
    printf("max |err| vs golden: %e (normalized)\r\n", max_err);
    printf(max_err < 1e-3 ? "PASS\r\n" : "FAIL\r\n");

    /* In deployment: at each 15-min tick read sensors, normalize with
     * norm_feat(), call pcnn_step_hw(). Re-warm-start with measured room
     * temperature at the start of every prediction horizon. */
    return 0;
}
