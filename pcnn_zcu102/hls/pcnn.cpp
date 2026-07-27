// PCNN single-zone accelerator -- Vitis HLS implementation.
// One invocation = one 15-minute timestep. State (LSTM h/c, D, E) is held in
// static on-chip memory across invocations, mirroring the recurrent use of
// the PyTorch model in pcnn_epfl (predict() calls the module step by step).
//
// Interface: AXI4-Lite (control + data). Weights are baked into BRAM/LUTROM
// from pcnn_weights.h at synthesis time (~1.4 MB floats, fits ZCU102 BRAM).

#include "pcnn.h"
#include "pcnn_weights.h"
#include <math.h>

#define PAR 8   // dot-product lanes (partial-sum accumulators)

static inline data_t sigm(data_t v) { return 1.0f / (1.0f + expf(-v)); }

// dot product with PAR partial sums so the pipeline reaches low II with floats
static data_t dot(const float *w, const data_t *x, int n) {
#pragma HLS INLINE off
    data_t ps[PAR];
#pragma HLS ARRAY_PARTITION variable=ps complete
    for (int i = 0; i < PAR; i++) ps[i] = 0.0f;
    for (int k = 0; k < n; k += PAR) {
#pragma HLS PIPELINE II=4
        for (int j = 0; j < PAR; j++) {
#pragma HLS UNROLL
            int idx = k + j;
            if (idx < n) ps[j] += w[idx] * x[idx];
        }
    }
    data_t s = 0.0f;
    for (int j = 0; j < PAR; j++) s += ps[j];
    return s;
}

extern "C" void pcnn_step(const data_t x[N_FEAT], int mode,
                          data_t *T_out, data_t *D_out, data_t *E_out) {
#pragma HLS INTERFACE s_axilite port=x      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=mode   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=T_out  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=D_out  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=E_out  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // ---- persistent state --------------------------------------------------
    static data_t h[LSTM_L][LSTM_H];
    static data_t c[LSTM_L][LSTM_H];
    static data_t last_D = 0.0f;
    static data_t last_E = 0.0f;

    if (mode == PCNN_MODE_RESET_WARM) {
        for (int l = 0; l < LSTM_L; l++)
            for (int i = 0; i < LSTM_H; i++) {
#pragma HLS PIPELINE II=1
                h[l][i] = INIT_H[l * LSTM_H + i];
                c[l][i] = INIT_C[l * LSTM_H + i];
            }
        last_D = 0.0f;
        last_E = 0.0f;
    }
    const bool warm = (mode != PCNN_MODE_PRED);

    // ---- temperature feedback (module.py: x[:,-1,temp] overwrite) ----------
    const data_t x_temp = warm ? (data_t)(x[COL_T] - last_E) : last_D;

    // ---- input MLP: 2 -> IN_NN, ReLU ---------------------------------------
    data_t xin[2];
    xin[0] = x[COL_SOL];
    xin[1] = x[COL_OCC];
    data_t emb[IN_NN];
    for (int i = 0; i < IN_NN; i++) {
        data_t a = W_IN[i * 2 + 0] * xin[0] + W_IN[i * 2 + 1] * xin[1] + B_IN[i];
        emb[i] = (a > 0.0f) ? a : 0.0f;
    }

    // ---- LSTM stack: 3 layers, hidden LSTM_H -------------------------------
    // PyTorch gate order in weights: [i, f, g, o]
    static const float *WIH[LSTM_L] = { LSTM_WIH0, LSTM_WIH1, LSTM_WIH2 };
    static const float *WHH[LSTM_L] = { LSTM_WHH0, LSTM_WHH1, LSTM_WHH2 };
    static const float *BIH[LSTM_L] = { LSTM_BIH0, LSTM_BIH1, LSTM_BIH2 };
    static const float *BHH[LSTM_L] = { LSTM_BHH0, LSTM_BHH1, LSTM_BHH2 };

    data_t layer_in[LSTM_H];   // max(IN_NN, LSTM_H)
    int in_sz = IN_NN;
    for (int i = 0; i < IN_NN; i++) layer_in[i] = emb[i];

    for (int l = 0; l < LSTM_L; l++) {
        data_t hn[LSTM_H], cn[LSTM_H];
        for (int u = 0; u < LSTM_H; u++) {
            data_t gi = BIH[l][0 * LSTM_H + u] + BHH[l][0 * LSTM_H + u]
                      + dot(&WIH[l][(0 * LSTM_H + u) * in_sz], layer_in, in_sz)
                      + dot(&WHH[l][(0 * LSTM_H + u) * LSTM_H], h[l], LSTM_H);
            data_t gf = BIH[l][1 * LSTM_H + u] + BHH[l][1 * LSTM_H + u]
                      + dot(&WIH[l][(1 * LSTM_H + u) * in_sz], layer_in, in_sz)
                      + dot(&WHH[l][(1 * LSTM_H + u) * LSTM_H], h[l], LSTM_H);
            data_t gg = BIH[l][2 * LSTM_H + u] + BHH[l][2 * LSTM_H + u]
                      + dot(&WIH[l][(2 * LSTM_H + u) * in_sz], layer_in, in_sz)
                      + dot(&WHH[l][(2 * LSTM_H + u) * LSTM_H], h[l], LSTM_H);
            data_t go = BIH[l][3 * LSTM_H + u] + BHH[l][3 * LSTM_H + u]
                      + dot(&WIH[l][(3 * LSTM_H + u) * in_sz], layer_in, in_sz)
                      + dot(&WHH[l][(3 * LSTM_H + u) * LSTM_H], h[l], LSTM_H);
            cn[u] = sigm(gf) * c[l][u] + sigm(gi) * tanhf(gg);
            hn[u] = sigm(go) * tanhf(cn[u]);
        }
        for (int u = 0; u < LSTM_H; u++) {
#pragma HLS PIPELINE II=1
            h[l][u] = hn[u];
            c[l][u] = cn[u];
            layer_in[u] = hn[u];
        }
        in_sz = LSTM_H;
    }

    // ---- layer norm --------------------------------------------------------
    data_t mean = 0.0f;
    for (int i = 0; i < LSTM_H; i++) mean += layer_in[i];
    mean /= (data_t)LSTM_H;
    data_t var = 0.0f;
    for (int i = 0; i < LSTM_H; i++) {
        data_t d = layer_in[i] - mean;
        var += d * d;
    }
    var /= (data_t)LSTM_H;
    const data_t inv_std = 1.0f / sqrtf(var + 1e-5f);
    data_t ln[LSTM_H];
    for (int i = 0; i < LSTM_H; i++)
        ln[i] = (layer_in[i] - mean) * inv_std * LN_G[i] + LN_B[i];

    // ---- output MLP: LSTM_H -> OUT_NN (tanh) -> 1 (tanh) -------------------
    data_t o1[OUT_NN];
    for (int i = 0; i < OUT_NN; i++)
        o1[i] = tanhf(dot(&W_OUT1[i * LSTM_H], ln, LSTM_H) + B_OUT1[i]);
    data_t o2 = tanhf(dot(W_OUT2, o1, OUT_NN) + B_OUT2[0]);

    // ---- D module (ResNet-like) -------------------------------------------
    const data_t D = o2 / DIV_FACTOR + x_temp;

    // ---- E module (physics) ------------------------------------------------
    // heat loss to outside: E -= b * (T_room - T_out)   (unnormalized temps)
    const data_t T_un  = (x_temp + last_E - 0.1f) / 0.8f * ROOM_DIFF + ROOM_MIN;
    const data_t To_un = (x[COL_OUT] - 0.1f) / 0.8f * OUT_DIFF + OUT_MIN;
    data_t E = last_E - B_EFF * (T_un - To_un);

    // HVAC effect: heating adds a*u, cooling adds d*u (u<0)
    const data_t p_eff = (x[COL_P] > 0.05f) ? x[COL_P] : (data_t)ZERO_POWER;
    const bool active = fabsf(p_eff - ZERO_POWER) > 1e-6f;
    if (active) {
        const data_t pw = x[COL_P] - ZERO_POWER;
        if (x[COL_CASE] > 0.5f)      E += A_EFF * pw;   // heating
        else if (x[COL_CASE] < 0.5f) E += D_EFF * pw;   // cooling
    }

    // ---- update state, write outputs --------------------------------------
    last_D = D;
    last_E = E;
    *T_out = D + E;
    *D_out = D;
    *E_out = E;
}
