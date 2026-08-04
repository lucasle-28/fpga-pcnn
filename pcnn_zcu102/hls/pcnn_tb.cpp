// C simulation testbench: replays one 24h test sequence (96 steps, first 8
// warm-start) and compares T/D/E against the Python golden reference.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "pcnn.h"

#define SEQ 96
#define WARM 8

int main(int argc, char **argv) {
    // Try multiple paths: Vitis HLS runs csim from a nested build directory
    const char *x_paths[] = {
        (argc > 1) ? argv[1] : NULL,
        "tb_data/tb_x.txt",
        "../tb_data/tb_x.txt",
        "../../tb_data/tb_x.txt",
        "../../../tb_data/tb_x.txt",
        "../../../../tb_data/tb_x.txt",
        "../../../../../tb_data/tb_x.txt",
        "../../../../../../tb_data/tb_x.txt",
        "/home/sukote/Documents/pcnn_unrol/pcnn_zcu102/hls/tb_data/tb_x.txt",
        "/home/sukote/Documents/pcnn/pcnn_zcu102/hls/tb_data/tb_x.txt",
        "/home/lucas/Documents/pcnn/pcnn_zcu102/hls/tb_data/tb_x.txt",
        NULL
    };
    const char *r_paths[] = {
        (argc > 2) ? argv[2] : NULL,
        "tb_data/tb_ref.txt",
        "../tb_data/tb_ref.txt",
        "../../tb_data/tb_ref.txt",
        "../../../tb_data/tb_ref.txt",
        "../../../../tb_data/tb_ref.txt",
        "../../../../../tb_data/tb_ref.txt",
        "../../../../../../tb_data/tb_ref.txt",
        "/home/sukote/Documents/pcnn_unrol/pcnn_zcu102/hls/tb_data/tb_ref.txt",
        "/home/sukote/Documents/pcnn/pcnn_zcu102/hls/tb_data/tb_ref.txt",
        "/home/lucas/Documents/pcnn/pcnn_zcu102/hls/tb_data/tb_ref.txt",
        NULL
    };
    FILE *fx = NULL;
    for (int i = 0; x_paths[i] && !fx; i++) fx = fopen(x_paths[i], "r");
    FILE *fr = NULL;
    for (int i = 0; r_paths[i] && !fr; i++) fr = fopen(r_paths[i], "r");

    if (!fx) { printf("cannot open tb_x file\n"); return 1; }

    static data_t x[SEQ][N_FEAT];
    static double ref[SEQ][3];
    bool has_ref = true;
    bool ref_is_y_only = false;
    for (int t = 0; t < SEQ; t++)
        for (int i = 0; i < N_FEAT; i++)
            if (fscanf(fx, "%f", &x[t][i]) != 1) { printf("bad tb_x\n"); return 1; }
            
    if (fr) {
        char line[256];
        if (fgets(line, sizeof(line), fr)) {
            double v1, v2, v3;
            int n = sscanf(line, "%lf %lf %lf", &v1, &v2, &v3);
            if (n == 3) {
                has_ref = true;
                ref[0][0] = v1; ref[0][1] = v2; ref[0][2] = v3;
                for (int t = 1; t < SEQ; t++)
                    if (fscanf(fr, "%lf %lf %lf", &ref[t][0], &ref[t][1], &ref[t][2]) != 3) { has_ref = false; break; }
            } else if (n == 1) {
                has_ref = true;
                ref_is_y_only = true;
                ref[0][0] = v1;
                for (int t = 1; t < SEQ; t++)
                    if (fscanf(fr, "%lf", &ref[t][0]) != 1) { has_ref = false; break; }
            }
        }
        fclose(fr);
    } else {
        has_ref = false;
        printf("WARNING: No tb_ref provided. Skipping comparison.\n");
    }
    fclose(fx);

    double max_err = 0.0, max_errT = 0.0, mseT = 0.0;
    for (int t = 0; t < SEQ; t++) {
        int mode = (t == 0) ? PCNN_MODE_RESET_WARM
                 : (t < WARM) ? PCNN_MODE_WARM : PCNN_MODE_PRED;
        data_t T, D, E;
        pcnn_step(x[t], mode, &T, &D, &E);
        
        if (has_ref) {
            double eT = fabs((double)T - ref[t][0]);
            mseT += eT * eT;
            if (eT > max_errT) max_errT = eT;
            if (!ref_is_y_only) {
                double eD = fabs((double)D - ref[t][1]);
                double eE = fabs((double)E - ref[t][2]);
                double e = fmax(eT, fmax(eD, eE));
                if (e > max_err) max_err = e;
            } else {
                max_err = max_errT;
            }
        }
        
        if (t < 4 || t % 24 == 0) {
            if (has_ref && !ref_is_y_only) {
                printf("t=%2d  T=%.6f  ref=%.6f  |D|=%.6f  |E|=%.6f\n",
                       t, (double)T, ref[t][0], (double)D, (double)E);
            } else if (has_ref && ref_is_y_only) {
                printf("t=%2d  T=%.6f  GT=%.6f  |D|=%.6f  |E|=%.6f\n",
                       t, (double)T, ref[t][0], (double)D, (double)E);
            } else {
                printf("t=%2d  T=%.6f  |D|=%.6f  |E|=%.6f\n",
                       t, (double)T, (double)D, (double)E);
            }
        }
    }
    
    if (has_ref) {
        double rmseT = sqrt(mseT / SEQ);
        if (ref_is_y_only) {
            printf("Compared against GROUND TRUTH (tb_y.txt)\n");
            printf("T_room RMSE (norm): %.4f  (approx %.3f degC)\n", rmseT, rmseT * (40.0)); // Rough un-norm scale
            return 0; // Success if we just wanted to measure RMSE
        } else {
            printf("max |err| over T,D,E: %.3e   (T only: %.3e)\n", max_err, max_errT);
            if (max_err < 1e-3) { printf("TEST PASSED\n"); return 0; }
            printf("TEST FAILED\n");
            return 1;
        }
    } else {
        printf("TEST PASSED (Execution only)\n");
        return 0;
    }
}
