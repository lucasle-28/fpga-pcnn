# PCNN — Physics-Constrained Neural Network on FPGA

A custom FPGA accelerator that runs a **Physics-Constrained Neural Network (PCNN)** on the Xilinx ZCU102 (Zynq UltraScale+ XCZU9EG) for real-time building thermal prediction. The accelerator predicts indoor room temperature 24 hours ahead at 15-minute resolution, combining a learned neural network (the "D" data-driven module) with first-principles thermodynamics (the "E" physics module) in a single HLS-synthesized IP core driven by a bare-metal Cortex-A53 host.

> **What makes this interesting:** the entire neural network — 3-layer LSTM with ~345k multiply-accumulates per timestep — runs as a deterministic hardware pipeline on the FPGA fabric with sub-millisecond latency per step, zero OS jitter, and bit-exact parity with the original PyTorch model. It has been validated both against golden model outputs and against real-world sensor data from Harvard's Science & Engineering Complex.

---

## How It Works

The PCNN architecture (from [PCNN-Comparison](https://github.com/XuezhengWang/PCNN-Comparison), Wang et al. 2026 Table 6) decomposes room temperature prediction into two parallel streams:

```
                         ┌──────────────────────────────────────────────┐
  Solar radiation ──┐    │              D Module (Data-driven)         │
  Occupancy ────────┼──▶ │  Input MLP(2→32, ReLU)                     │
                    │    │    → LSTM(3 layers × 128 hidden)            │──▶ D_k+1 = nn/30 + T̃_k
                    │    │    → LayerNorm(128)                         │
                    │    │    → Output MLP(128→32→1, tanh)             │
                    │    └──────────────────────────────────────────────┘
                    │                                                        T = D + E
  Outdoor temp ─────┤    ┌──────────────────────────────────────────────┐
  Room temp ────────┤    │              E Module (Physics)             │
  HVAC power ───────┤    │  Heat loss:  E -= b·(T_room - T_outdoor)   │──▶ E_k+1
  Heating/cooling ──┘    │  HVAC gain:  E += a·u (heat) or d·u (cool) │
                         └──────────────────────────────────────────────┘
```

One accelerator invocation = one 15-minute timestep. The LSTM hidden/cell states and D/E accumulators persist in on-chip BRAM between invocations, exactly mirroring the recurrent step-by-step inference of `Model.predict()` from the original PyTorch codebase.

Each 24-hour prediction horizon consists of **96 steps**:
- Steps 0–7 (**warm-start**): feed true measured room temperature to condition the LSTM states
- Steps 8–95 (**autoregressive**): the model feeds its own predictions back, forecasting up to 22 hours ahead

---

## Project Layout

```
├── harvard_to_pcnn.py              # Converts Harvard SEC sensor data → PCNN test vectors
├── Mid.csv                         # EnergyPlus Mid building training dataset
├── data_2024-04-15_...csv          # Harvard SEC real-world BMS export (14 months)
│
├── pcnn_zcu102/
│   ├── hls/
│   │   ├── pcnn.h                  # HLS top-function header
│   │   ├── pcnn.cpp                # HLS accelerator (sigmoid LUT, PAR=4 dot products)
│   │   ├── pcnn_tb.cpp             # C-sim testbench (golden ref or ground-truth RMSE)
│   │   ├── run_hls.tcl             # Vitis HLS build script (100 MHz, xczu9eg)
│   │   ├── weights/
│   │   │   ├── pcnn_weights.h      # Baked weights (~6.4 MB, all in BRAM/LUTROM)
│   │   │   ├── pcnn_config.h       # Model dimensions, physics constants, normalization
│   │   │   └── sigm_lut256.inc     # 256-entry sigmoid lookup table
│   │   └── tb_data/
│   │       ├── tb_x.txt            # 96-step test inputs (normalized)
│   │       ├── tb_ref.txt          # Golden T, D, E per step (from PyTorch)
│   │       └── tb_y.txt            # Ground-truth room temperature
│   │
│   ├── python/
│   │   ├── pcnn_jax.py             # JAX reimplementation (train demo weights + export)
│   │   ├── export_weights_pytorch.py  # Export real PyTorch checkpoint → HLS headers
│   │   └── make_harvard_header.py  # Convert test vectors → C header for bare-metal app
│   │
│   └── vitis/
│       ├── main.c                  # Bare-metal host app (Cortex-A53, AXI4-Lite driver)
│       ├── pcnn_config.h           # Copy of config for host-side normalization
│       └── demo_sequence.h         # Embedded test data (generated, ~2.4 MB)
│
├── out/                            # Generated test vectors (per-room)
│   ├── 5.417_tb_x.txt              # Harvard room 5.417: inputs (18k+ steps)
│   ├── 5.417_tb_y.txt              # Harvard room 5.417: ground truth
│   └── norm_constants.json         # RAW_MIN/RAW_MAX used for normalization
│
└── photos/                         # Build & results screenshots
```

---

## Accelerator Interface

```c
void pcnn_step(const float x[6], int mode,
               float *T_out, float *D_out, float *E_out);
```

| Port | Description |
|------|-------------|
| `x[6]` | Normalized features: `[T_outdoor, solar, occupancy, T_room, P_hvac, case]`, min-max scaled to [0.1, 0.9] |
| `mode` | `0` = reset state + warm-start, `1` = warm-start, `2` = autoregressive prediction |
| `T_out` | Predicted room temperature (normalized) = D + E |
| `D_out` | Data-driven component |
| `E_out` | Physics component |

All ports on a single **AXI4-Lite** bundle (`ap_ctrl_hs` block-level protocol). The host writes 6 floats + mode, starts the accelerator, polls `ap_done`, then reads back T/D/E.

---

## FPGA Resource Utilization (Post-Implementation)

Achieved at **100 MHz** on the XCZU9EG with all timing constraints met (WNS = 2.625 ns):

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT      | 74,988 | 274,080 | 27.36% |
| LUTRAM   | 2,047 | 144,000 | 1.42% |
| FF       | 97,990 | 548,160 | 17.88% |
| BRAM     | 385 | 912 | 42.21% |
| DSP      | 712 | 2,520 | 28.25% |

BRAM is the bottleneck (~42%) because the ~1.4M float32 weights are stored entirely on-chip. DSP usage (~28%) is managed via `HLS ALLOCATION` pragmas that cap `fmul` at 650 and `fadd` at 600 instances, plus a LUT-based sigmoid that replaces expensive `expf` cores.

---

## Results

### Bit-Exact Verification (EnergyPlus Golden Model)

C-simulation against PyTorch golden reference over a 96-step horizon:

```
max |err| vs golden: 1.788139e-07 (normalized)
TEST PASSED
```

The on-board bare-metal run reproduces the same result — max error ~4e-05 in normalized units (~0.001 °C), confirming bit-consistency from PyTorch → HLS C-sim → synthesized RTL → on-chip execution.

### Real-World Validation (Harvard SEC Dataset)

Tested against 14 months of real building sensor data from Harvard's Science & Engineering Complex (room 5.417). The model was trained on synthetic EnergyPlus data (Mid.csv) and evaluated zero-shot on real-world data:

```
=== Overall: 195 horizons, 17160 prediction steps ===
RMSE (normalized): 0.132386
RMSE (degC):       3.461
```

Per-horizon RMSE ranges from **~0.9 °C** (best) to **~6.6 °C** (worst), reflecting the domain gap between simulated training data and real building dynamics. This validates the accelerator runs correctly on real-world data and provides a baseline for transfer learning.

---

## Build Flow

### 1. Weight Export

**Option A — Real PyTorch checkpoint** (recommended):
```bash
# From the PCNN-Comparison repo root:
python3 pcnn_zcu102/python/export_weights_pytorch.py \
    --ckpt saves/models/.../best_model.pt \
    --csv "Processed E+ data/Mid.csv" \
    --out pcnn_zcu102/hls/weights
```

**Option B — JAX demo weights** (no PyTorch needed):
```bash
python3 pcnn_zcu102/python/pcnn_jax.py --csv Mid.csv --train-seconds 120
python3 pcnn_zcu102/python/pcnn_jax.py --csv Mid.csv --export pcnn_zcu102/hls/weights
```

Both paths generate `pcnn_weights.h`, `pcnn_config.h`, and golden test vectors (`tb_x.txt`, `tb_ref.txt`).

### 2. Harvard Dataset Testing (No Xilinx Tools Required)

You can validate the accelerator against real-world sensor data using only a standard C++ compiler:

```bash
# Generate test vectors from Harvard BMS data
./harvard_to_pcnn.py --harvard "data_...csv" --mid "Mid.csv" \
    --config pcnn_zcu102/hls/weights/pcnn_config.h --outdir ./out --rooms 5.417

# Compile and run with native GCC
cd pcnn_zcu102/hls
g++ -I./weights -I. pcnn.cpp pcnn_tb.cpp -o pcnn_sim
./pcnn_sim ../../out/5.417_tb_x.txt ../../out/5.417_tb_y.txt
```

The testbench auto-detects 1-column `tb_y.txt` (ground truth) vs 3-column `tb_ref.txt` (golden model) and reports either RMSE or max absolute error accordingly.

### 3. HLS Synthesis

```bash
cd pcnn_zcu102/hls
vitis_hls -f run_hls.tcl
```

Runs C-simulation (must print `TEST PASSED`), synthesis at 100 MHz, and exports the IP to `pcnn_hls/sol1/impl/ip`.

### 4. Vivado Block Design (2022.x+)

1. New project targeting the ZCU102 board
2. Create block design → add **Zynq UltraScale+ MPSoC** → run block automation
3. Enable one PS-PL AXI master port (M_AXI_HPM0_FPD or LPD)
4. Add IP repository → `hls/pcnn_hls/sol1/impl/ip` → drop **Pcnn_step** into the design
5. Run connection automation (SmartConnect, `pl_clk0` 100 MHz, `pl_resetn0`)
6. Validate → create HDL wrapper → generate bitstream → **Export Hardware** (include bitstream) → `pcnn_zcu102.xsa`

### 5. Bare-Metal Application (Vitis)

1. Create platform project from `pcnn_zcu102.xsa` (standalone, psu_cortexa53_0)
2. Create application project; add `vitis/main.c`, `pcnn_config.h`, and `demo_sequence.h`:
   ```bash
   # Generate the embedded test header (e.g., all 195 horizons from Harvard data):
   python3 pcnn_zcu102/python/make_harvard_header.py \
       out/5.417_tb_x.txt out/5.417_tb_y.txt --horizons 195 \
       > pcnn_zcu102/vitis/demo_sequence.h
   ```
3. The HLS-generated driver (`xpcnn_step.*`) is auto-included by the BSP
4. Board setup: SW6 → JTAG boot, USB-UART at 115200 baud
5. Run/Debug from Vitis — UART prints per-step predictions, per-horizon RMSE, and overall accuracy

---

## Verification Chain

```
JAX/PyTorch golden model ──▶ tb_ref.txt ──▶ HLS C-sim (pcnn_tb.cpp)
       ──▶ (optional) cosim_design RTL vs C ──▶ on-board check in main.c
```

All stages compare T, D, E with tolerance 1e-3 in normalized units (~0.026 °C for this dataset). The real-world Harvard path uses ground-truth RMSE instead of golden comparison.

---

## Design Decisions & Notes

- **Float32 throughout**: `data_t` is `float` for exact parity with PyTorch. For a smaller/faster design, switch to `ap_fixed<24,8>` and re-run C-sim to measure quantization error (the physics module constants span ~1e-3 to 1e+2, so keep ≥ 16 fractional bits).

- **LUT-based sigmoid**: A 256-entry lookup table with linear interpolation replaces the expensive HLS `expf` polynomial core (which consumes ~14 DSPs per instance). `tanhf` still uses the HLS built-in (5 DSP, cheaper than a second LUT).

- **PAR=4 dot-product lanes**: Reduced from the original PAR=16 to fit the ZCU102's 2,520 DSP budget after synthesis showed over-utilization. Each lane computes partial sums in parallel with `ARRAY_PARTITION` and `PIPELINE` pragmas.

- **DSP budget caps**: `#pragma HLS ALLOCATION` limits `fmul` to 650 and `fadd` to 600 instances, preventing HLS from instantiating more floating-point operators than the device can support.

- **Occupancy remapping**: Harvard BMS provides binary occupancy (0/1); the model was trained on EnergyPlus headcounts. The converter maps binary → the modal values from Mid.csv (0.726/3.0), which happen to be exactly `RAW_MIN[2]`/`RAW_MAX[2]`, producing clean 0.1/0.9 normalized values.

- **Multi-zone extension**: The single-zone E module can be duplicated per zone with a wider output MLP for S-PCNN/M-PCNN variants; the weight export scripts extend directly.

---

## References

- Wang et al. (2026), *"Dual-phase evaluation framework for physics-constrained neural networks"* — Table 6 for model hyperparameters
- Di Natale et al. (EPFL), PCNN architecture — [PCNN-Comparison](https://github.com/XuezhengWang/PCNN-Comparison)
- Harvard SEC dataset — real-world BMS data from the Science & Engineering Complex
- Xilinx ZCU102 — Zynq UltraScale+ XCZU9EG-FFVB1156-2-E
