// Test harness: run PCNN with different sigmoid/tanh approximations
// and report error vs golden reference.
// Compile: g++ -O2 -Iweights -I. -o test_approx test_approx.cpp -lm

#include "pcnn.h"
#include "weights/pcnn_weights.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ========== Approximation candidates ==========

// Option A: Hard sigmoid / hard tanh (cheapest, ~1 fmul each)
static inline data_t sigm_hard(data_t v) {
    data_t y = v * 0.16666667f + 0.5f;
    if (y <= 0.0f) return 0.0f;
    if (y >= 1.0f) return 1.0f;
    return y;
}
static inline data_t tanh_hard(data_t v) {
    if (v <= -3.0f) return -1.0f;
    if (v >= 3.0f)  return 1.0f;
    return v * 0.33333333f;
}

// Option B: 3rd-order polynomial (better accuracy, ~3 fmul each)
static inline data_t sigm_poly3(data_t v) {
    if (v >= 4.0f) return 1.0f;
    if (v <= -4.0f) return 0.0f;
    // Chebyshev-style fit over [-4,4]
    data_t x2 = v * v;
    return 0.5f + v * (0.2f - 0.0045f * x2);
}
static inline data_t tanh_poly3(data_t v) {
    if (v >= 4.0f) return 1.0f;
    if (v <= -4.0f) return -1.0f;
    data_t x2 = v * v;
    return v * (1.0f - x2 * 0.1f + x2 * x2 * 0.005f);
}

// Option C: PLAN approximation (piecewise linear, 4 segments)
static inline data_t sigm_plan(data_t v) {
    data_t a = (v < 0.0f) ? -v : v;
    data_t y;
    if      (a >= 5.0f) y = 1.0f;
    else if (a >= 2.375f) y = 0.03125f * a + 0.84375f;
    else if (a >= 1.0f)   y = 0.125f   * a + 0.625f;
    else                  y = 0.25f    * a + 0.5f;
    return (v >= 0.0f) ? y : 1.0f - y;
}
static inline data_t tanh_plan(data_t v) {
    return 2.0f * sigm_plan(2.0f * v) - 1.0f;
}

// Option D: Fine PLAN (8-segment piecewise linear sigmoid)
static inline data_t sigm_plan8(data_t v) {
    data_t a = (v < 0.0f) ? -v : v;
    data_t y;
    if      (a >= 5.0f)    y = 1.0f;
    else if (a >= 3.75f)   y = 0.016f * a + 0.920f;
    else if (a >= 2.5f)    y = 0.048f * a + 0.800f;
    else if (a >= 1.75f)   y = 0.088f * a + 0.700f;
    else if (a >= 1.0f)    y = 0.128f * a + 0.630f;
    else if (a >= 0.5f)    y = 0.185f * a + 0.573f;
    else if (a >= 0.25f)   y = 0.225f * a + 0.553f;
    else                   y = 0.245f * a + 0.500f;
    return (v >= 0.0f) ? y : 1.0f - y;
}
static inline data_t tanh_plan8(data_t v) {
    return 2.0f * sigm_plan8(2.0f * v) - 1.0f;
}

// Option E: LUT with linear interpolation (256 entries, range [-8,8])
#define LUT_XMIN -8.0f
#define LUT_XMAX  8.0f

// We test multiple LUT sizes
#define LUT_N1 512
#define LUT_DX1 ((LUT_XMAX - LUT_XMIN) / LUT_N1)
static float sigm_lut1[LUT_N1 + 1];
static float tanh_lut1[LUT_N1 + 1];

#define LUT_N2 1024
#define LUT_DX2 ((LUT_XMAX - LUT_XMIN) / LUT_N2)
static float sigm_lut2[LUT_N2 + 1];
static float tanh_lut2[LUT_N2 + 1];

static bool lut_init_done = false;

static void init_lut() {
    if (lut_init_done) return;
    for (int i = 0; i <= LUT_N1; i++) {
        float x = LUT_XMIN + i * LUT_DX1;
        sigm_lut1[i] = 1.0f / (1.0f + expf(-x));
        tanh_lut1[i] = tanhf(x);
    }
    for (int i = 0; i <= LUT_N2; i++) {
        float x = LUT_XMIN + i * LUT_DX2;
        sigm_lut2[i] = 1.0f / (1.0f + expf(-x));
        tanh_lut2[i] = tanhf(x);
    }
    lut_init_done = true;
}

// 512-entry LUT
static inline data_t sigm_lut512(data_t v) {
    if (v <= LUT_XMIN) return 0.0f;
    if (v >= LUT_XMAX) return 1.0f;
    float fi = (v - LUT_XMIN) / LUT_DX1;
    int i = (int)fi;
    float frac = fi - i;
    return sigm_lut1[i] + frac * (sigm_lut1[i+1] - sigm_lut1[i]);
}
static inline data_t tanh_lut512(data_t v) {
    if (v <= LUT_XMIN) return -1.0f;
    if (v >= LUT_XMAX) return 1.0f;
    float fi = (v - LUT_XMIN) / LUT_DX1;
    int i = (int)fi;
    float frac = fi - i;
    return tanh_lut1[i] + frac * (tanh_lut1[i+1] - tanh_lut1[i]);
}

// 1024-entry LUT
static inline data_t sigm_lut1024(data_t v) {
    if (v <= LUT_XMIN) return 0.0f;
    if (v >= LUT_XMAX) return 1.0f;
    float fi = (v - LUT_XMIN) / LUT_DX2;
    int i = (int)fi;
    float frac = fi - i;
    return sigm_lut2[i] + frac * (sigm_lut2[i+1] - sigm_lut2[i]);
}
static inline data_t tanh_lut1024(data_t v) {
    if (v <= LUT_XMIN) return -1.0f;
    if (v >= LUT_XMAX) return 1.0f;
    float fi = (v - LUT_XMIN) / LUT_DX2;
    int i = (int)fi;
    float frac = fi - i;
    return tanh_lut2[i] + frac * (tanh_lut2[i+1] - tanh_lut2[i]);
}

// Exact (reference)
static inline data_t sigm_exact(data_t v) { return 1.0f / (1.0f + expf(-v)); }
static inline data_t tanh_exact(data_t v) { return tanhf(v); }

// ========== PCNN engine (parameterized by sigmoid/tanh) ==========

#define PAR_TEST 2  // Match current PAR

static data_t dot_t(const float *w, const data_t *x, int n) {
    data_t ps[PAR_TEST] = {};
    for (int k = 0; k < n; k += PAR_TEST)
        for (int j = 0; j < PAR_TEST; j++) {
            int idx = k + j;
            if (idx < n) ps[j] += w[idx] * x[idx];
        }
    data_t s = 0.0f;
    for (int j = 0; j < PAR_TEST; j++) s += ps[j];
    return s;
}

typedef data_t (*act_fn)(data_t);

static void pcnn_run(const data_t input[][N_FEAT], int nsteps, int warm_steps,
                     act_fn sig, act_fn tnh,
                     double *out_T, double *out_D, double *out_E) {
    static data_t h[LSTM_L][LSTM_H], c[LSTM_L][LSTM_H];
    static data_t last_D, last_E;

    for (int t = 0; t < nsteps; t++) {
        int mode = (t == 0) ? PCNN_MODE_RESET_WARM
                 : (t < warm_steps) ? PCNN_MODE_WARM : PCNN_MODE_PRED;

        if (mode == PCNN_MODE_RESET_WARM) {
            for (int l = 0; l < LSTM_L; l++)
                for (int i = 0; i < LSTM_H; i++) {
                    h[l][i] = INIT_H[l * LSTM_H + i];
                    c[l][i] = INIT_C[l * LSTM_H + i];
                }
            last_D = 0.0f; last_E = 0.0f;
        }
        bool warm = (mode != PCNN_MODE_PRED);
        data_t x_temp = warm ? (data_t)(input[t][COL_T] - last_E) : last_D;

        // input MLP
        data_t xin[2] = { input[t][COL_SOL], input[t][COL_OCC] };
        data_t emb[IN_NN];
        for (int i = 0; i < IN_NN; i++) {
            data_t a = W_IN[i*2+0]*xin[0] + W_IN[i*2+1]*xin[1] + B_IN[i];
            emb[i] = (a > 0.0f) ? a : 0.0f;
        }

        // LSTM
        data_t layer_in[LSTM_H];
        int in_sz = IN_NN;
        for (int i = 0; i < IN_NN; i++) layer_in[i] = emb[i];

        for (int l = 0; l < LSTM_L; l++) {
            const float *wih = (l==0)?LSTM_WIH0:(l==1)?LSTM_WIH1:LSTM_WIH2;
            const float *whh = (l==0)?LSTM_WHH0:(l==1)?LSTM_WHH1:LSTM_WHH2;
            const float *bih = (l==0)?LSTM_BIH0:(l==1)?LSTM_BIH1:LSTM_BIH2;
            const float *bhh = (l==0)?LSTM_BHH0:(l==1)?LSTM_BHH1:LSTM_BHH2;
            data_t hn[LSTM_H], cn[LSTM_H];
            for (int u = 0; u < LSTM_H; u++) {
                data_t gi = bih[0*LSTM_H+u]+bhh[0*LSTM_H+u]
                    +dot_t(&wih[(0*LSTM_H+u)*in_sz],layer_in,in_sz)
                    +dot_t(&whh[(0*LSTM_H+u)*LSTM_H],h[l],LSTM_H);
                data_t gf = bih[1*LSTM_H+u]+bhh[1*LSTM_H+u]
                    +dot_t(&wih[(1*LSTM_H+u)*in_sz],layer_in,in_sz)
                    +dot_t(&whh[(1*LSTM_H+u)*LSTM_H],h[l],LSTM_H);
                data_t gg = bih[2*LSTM_H+u]+bhh[2*LSTM_H+u]
                    +dot_t(&wih[(2*LSTM_H+u)*in_sz],layer_in,in_sz)
                    +dot_t(&whh[(2*LSTM_H+u)*LSTM_H],h[l],LSTM_H);
                data_t go = bih[3*LSTM_H+u]+bhh[3*LSTM_H+u]
                    +dot_t(&wih[(3*LSTM_H+u)*in_sz],layer_in,in_sz)
                    +dot_t(&whh[(3*LSTM_H+u)*LSTM_H],h[l],LSTM_H);
                cn[u] = sig(gf)*c[l][u] + sig(gi)*tnh(gg);
                hn[u] = sig(go)*tnh(cn[u]);
            }
            for (int u = 0; u < LSTM_H; u++) {
                h[l][u]=hn[u]; c[l][u]=cn[u]; layer_in[u]=hn[u];
            }
            in_sz = LSTM_H;
        }

        // LayerNorm
        data_t mean=0; for(int i=0;i<LSTM_H;i++) mean+=layer_in[i];
        mean /= (data_t)LSTM_H;
        data_t var=0; for(int i=0;i<LSTM_H;i++){data_t d=layer_in[i]-mean; var+=d*d;}
        var /= (data_t)LSTM_H;
        data_t inv_std = 1.0f / sqrtf(var + 1e-5f);
        data_t ln[LSTM_H];
        for(int i=0;i<LSTM_H;i++) ln[i]=(layer_in[i]-mean)*inv_std*LN_G[i]+LN_B[i];

        // Output MLP
        data_t o1[OUT_NN];
        for(int i=0;i<OUT_NN;i++) o1[i]=tnh(dot_t(&W_OUT1[i*LSTM_H],ln,LSTM_H)+B_OUT1[i]);
        data_t o2 = tnh(dot_t(W_OUT2,o1,OUT_NN)+B_OUT2[0]);

        // D module
        data_t D = o2 / DIV_FACTOR + x_temp;

        // E module
        data_t T_un  = (x_temp + last_E - 0.1f)/0.8f*ROOM_DIFF+ROOM_MIN;
        data_t To_un = (input[t][COL_OUT]-0.1f)/0.8f*OUT_DIFF+OUT_MIN;
        data_t E = last_E - B_EFF*(T_un - To_un);
        data_t p_eff = (input[t][COL_P]>0.05f)?input[t][COL_P]:(data_t)ZERO_POWER;
        if (fabsf(p_eff - ZERO_POWER) > 1e-6f) {
            data_t pw = input[t][COL_P] - ZERO_POWER;
            if (input[t][COL_CASE]>0.5f) E+=A_EFF*pw;
            else if (input[t][COL_CASE]<0.5f) E+=D_EFF*pw;
        }

        last_D = D; last_E = E;
        out_T[t] = D+E; out_D[t] = D; out_E[t] = E;
    }
}

#define SEQ 96
#define WARM 8

int main() {
    FILE *fx = fopen("tb_data/tb_x.txt","r");
    FILE *fr = fopen("tb_data/tb_ref.txt","r");
    if (!fx) { printf("cannot open tb_x\n"); return 1; }

    static data_t x[SEQ][N_FEAT];
    static double ref[SEQ][3];
    for (int t=0;t<SEQ;t++) for(int i=0;i<N_FEAT;i++) fscanf(fx,"%f",&x[t][i]);
    for (int t=0;t<SEQ;t++) fscanf(fr,"%lf%lf%lf",&ref[t][0],&ref[t][1],&ref[t][2]);
    fclose(fx); fclose(fr);

    init_lut();

    struct { const char *name; act_fn sig; act_fn tnh; } opts[] = {
        {"Exact (expf/tanhf)",  sigm_exact, tanh_exact},
        {"Hard sigmoid/tanh",   sigm_hard,  tanh_hard},
        {"Poly3",               sigm_poly3, tanh_poly3},
        {"PLAN (4-segment)",    sigm_plan,  tanh_plan},
        {"PLAN (8-segment)",    sigm_plan8, tanh_plan8},
        {"LUT-512 interp",      sigm_lut512,  tanh_lut512},
        {"LUT-1024 interp",     sigm_lut1024, tanh_lut1024},
    };

    for (auto &o : opts) {
        double T[SEQ], D[SEQ], E[SEQ];
        pcnn_run(x, SEQ, WARM, o.sig, o.tnh, T, D, E);
        double maxe=0, maxeT=0;
        for (int t=0;t<SEQ;t++) {
            double eT=fabs(T[t]-ref[t][0]), eD=fabs(D[t]-ref[t][1]), eE=fabs(E[t]-ref[t][2]);
            double e=fmax(eT,fmax(eD,eE));
            if(e>maxe) maxe=e;
            if(eT>maxeT) maxeT=eT;
        }
        printf("%-22s  max|err|=%.3e  max|errT|=%.3e  %s\n",
               o.name, maxe, maxeT, (maxe<1e-3)?"PASS":"FAIL");
    }
}
