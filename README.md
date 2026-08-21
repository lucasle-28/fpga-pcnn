# PCNN — Physics-Constrained Neural Network on FPGA

A Physics-Constrained Neural Network running as a custom HLS accelerator on the Xilinx **ZCU102** (Zynq UltraScale+ XCZU9EG). It predicts indoor room temperature 24 h ahead at 15-minute resolution by combining a 3-layer LSTM (the **D**, data-driven module) with first-principles thermodynamics (the **E**, physics module). One accelerator call = one 15-minute step; a bare-metal Cortex-A53 host drives it over AXI4-Lite.

Trained on synthetic EnergyPlus data, verified bit-consistent from PyTorch → HLS C-sim → synthesized RTL → on-chip, then evaluated zero-shot on 14 months of real Harvard SEC building sensor data.

| | |
|---|---|
| **Golden-model agreement** | max abs err **4.76e-05** (normalized) — C-sim and on-board agree to the digit |
| **Real-world accuracy** | RMSE **3.461 °C** over 195 horizons / 17,160 steps (zero-shot, Harvard SEC) |
| **Resources @ 100 MHz** | LUT 27.4% · BRAM 42.2% · DSP 28.3% · WNS +2.625 ns |

---

## Quickstart — run it in 2 minutes, no Xilinx tools

The accelerator is plain C++ and compiles with any GCC. This is the fastest way to confirm the repo works:

```bash
cd pcnn_zcu102/hls
g++ -O2 -I./weights -I. pcnn.cpp pcnn_tb.cpp -o pcnn_sim

./pcnn_sim                    # golden-model check  -> "TEST PASSED"
./pcnn_sim ../../out/5.417_tb_x.txt ../../out/5.417_tb_y.txt   # real Harvard data -> RMSE
```

Expected output: `max |err| over T,D,E: 4.762e-05` / `TEST PASSED`, and `RMSE (norm): 0.1211 (approx 3.167 degC)`.

Python scripts (test-vector generation only): `pip install -r requirements.txt`.

Full board flow — HLS synthesis, Vivado block design, Vitis app — is in **[docs/build.md](docs/build.md)**.

---

## Repo map

```
├── README.md                       # you are here
├── requirements.txt                # numpy + pandas (test-vector scripts)
├── docs/                           # architecture, build flow, results
├── harvard_to_pcnn.py              # Harvard SEC BMS export -> PCNN test vectors
├── Mid.csv                         # EnergyPlus "Mid" building training dataset
├── data_2024-04-15_...csv          # Harvard SEC raw BMS export (14 months)
│
├── pcnn_zcu102/
│   ├── hls/
│   │   ├── pcnn.h / pcnn.cpp       # the accelerator (sigmoid LUT, PAR=4 dot products)
│   │   ├── pcnn_tb.cpp             # C-sim testbench (golden ref or ground-truth RMSE)
│   │   ├── run_hls.tcl             # Vitis HLS build script (100 MHz, xczu9eg)
│   │   ├── weights/
│   │   │   ├── pcnn_weights.h      # baked weights, ~6.4 MB, all on-chip
│   │   │   ├── pcnn_config.h       # dimensions, physics constants, normalization
│   │   │   └── sigm_lut256.inc     # 256-entry sigmoid table
│   │   └── tb_data/                # tb_x.txt (inputs), tb_ref.txt (golden T/D/E), tb_y.txt
│   │
│   ├── python/                     # weight export + header generation (see below)
│   └── vitis/
│       ├── main.c                  # bare-metal host app (Cortex-A53, AXI4-Lite)
│       ├── pcnn_config.h           # copy of weights/pcnn_config.h — keep in sync
│       └── demo_sequence.h         # generated, ~2.4 MB embedded test data
│
├── out/                            # generated Harvard test vectors (room 5.417)
└── photos/                         # build & results screenshots
```

## What generates what

Most confusion in this repo comes from generated files. This is the whole graph:

| Generated file | Produced by | From |
|---|---|---|
| `hls/weights/pcnn_weights.h`, `pcnn_config.h`, `tb_data/*` | `python/export_weights_pytorch.py` (real ckpt) **or** `python/pcnn_jax.py` (demo weights) | PyTorch checkpoint / `Mid.csv` |
| `hls/weights/sigm_lut256.inc` | `python/pcnn_jax.py --export` | — |
| `out/5.417_tb_x.txt`, `_tb_y.txt`, `norm_constants.json` | `harvard_to_pcnn.py` | Harvard CSV + `Mid.csv` + `pcnn_config.h` |
| `vitis/demo_sequence.h` | `python/make_harvard_header.py` | `out/5.417_tb_*.txt` |
| `vitis/pcnn_config.h` | copied by hand | `hls/weights/pcnn_config.h` |

Everything else is hand-written source. Regenerating weights invalidates `tb_data/` and `demo_sequence.h` — regenerate both.

---

## Accelerator interface

```c
void pcnn_step(const float x[6], int mode, float *T_out, float *D_out, float *E_out);
```

| Port | Meaning |
|---|---|
| `x[6]` | `[T_outdoor, solar, occupancy, T_room, P_hvac, case]`, min-max scaled to `[0.1, 0.9]` |
| `mode` | `0` = reset + warm-start · `1` = warm-start · `2` = autoregressive prediction |
| `T/D/E_out` | Predicted temperature (normalized) `T = D + E`, plus each component |

Single AXI4-Lite bundle, `ap_ctrl_hs`. LSTM hidden/cell state and the D/E accumulators live in on-chip BRAM across calls. A 24 h horizon = 96 steps: 0–7 warm-start on measured temperature, 8–95 autoregressive.

---

## Where to go next

- **[docs/architecture.md](docs/architecture.md)** — how the D and E modules work, and every HLS design decision (why a sigmoid LUT, why PAR=4, why float32) with its reasoning.
- **[docs/build.md](docs/build.md)** — weight export → HLS synthesis → Vivado block design → Vitis bare-metal app.
- **[docs/results.md](docs/results.md)** — verification chain, measured accuracy, resource utilization, screenshots.

## Known gaps

- **Per-step latency is not measured.** The design is a deterministic pipeline with no OS jitter, but neither `main.c` nor the docs record actual cycles. Add a `XTime_GetTime()` pair around the `pcnn_step` call in `main.c` to get a real number.
- **`vitis/pcnn_config.h` is a hand-copy** of the HLS one and can silently drift. They are identical today; diff them after any weight regeneration.
- **Real-world RMSE (3.46 °C) reflects a domain gap**, not an implementation bug — the model was trained on synthetic EnergyPlus data and never fine-tuned on Harvard data. Transfer learning is the obvious next step.
- **RTL co-simulation has never been run** (`cosim_design` is commented out in `run_hls.tcl`; it is very slow with this much state). Confidence in RTL correctness comes from the on-board run instead.

## References

- Wang et al. (2026), *Dual-phase evaluation framework for physics-constrained neural networks* — Table 6 for hyperparameters
- Di Natale et al. (EPFL), PCNN architecture — [PCNN-Comparison](https://github.com/XuezhengWang/PCNN-Comparison)
- Harvard SEC — real-world BMS data, Science & Engineering Complex
- Xilinx ZCU102 — Zynq UltraScale+ XCZU9EG-FFVB1156-2-E

---

Maintainer: Lucas &lt;lucasle5141@gmail.com&gt;
