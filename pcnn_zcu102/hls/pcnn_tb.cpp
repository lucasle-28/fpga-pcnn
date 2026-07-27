// C simulation testbench: replays one 24h test sequence (96 steps, first 8
// warm-start) and compares T/D/E against the Python golden reference.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "pcnn.h"

#define SEQ 96
#define WARM 8

int main(int argc, char **argv) {
    const char *xf = (argc > 1) ? argv[1] : "tb_data/tb_x.txt";
    const char *rf = (argc > 2) ? argv[2] : "tb_data/tb_ref.txt";
    FILE *fx = fopen(xf, "r");
    FILE *fr = fopen(rf, "r");
    if (!fx || !fr) { printf("cannot open tb_data files\n"); return 1; }

    static data_t x[SEQ][N_FEAT];
    static double ref[SEQ][3];
    for (int t = 0; t < SEQ; t++)
        for (int i = 0; i < N_FEAT; i++)
            if (fscanf(fx, "%f", &x[t][i]) != 1) { printf("bad tb_x\n"); return 1; }
    for (int t = 0; t < SEQ; t++)
        for (int i = 0; i < 3; i++)
            if (fscanf(fr, "%lf", &ref[t][i]) != 1) { printf("bad tb_ref\n"); return 1; }
    fclose(fx); fclose(fr);

    double max_err = 0.0, max_errT = 0.0;
    for (int t = 0; t < SEQ; t++) {
        int mode = (t == 0) ? PCNN_MODE_RESET_WARM
                 : (t < WARM) ? PCNN_MODE_WARM : PCNN_MODE_PRED;
        data_t T, D, E;
        pcnn_step(x[t], mode, &T, &D, &E);
        double eT = fabs((double)T - ref[t][0]);
        double eD = fabs((double)D - ref[t][1]);
        double eE = fabs((double)E - ref[t][2]);
        double e = fmax(eT, fmax(eD, eE));
        if (e > max_err) max_err = e;
        if (eT > max_errT) max_errT = eT;
        if (t < 4 || t % 24 == 0)
            printf("t=%2d  T=%.6f  ref=%.6f  |D|=%.6f  |E|=%.6f\n",
                   t, (double)T, ref[t][0], (double)D, (double)E);
    }
    printf("max |err| over T,D,E: %.3e   (T only: %.3e)\n", max_err, max_errT);
    // normalized units; 1e-3 corresponds to ~0.026 degC for the Mid dataset
    if (max_err < 1e-3) { printf("TEST PASSED\n"); return 0; }
    printf("TEST FAILED\n");
    return 1;
}
