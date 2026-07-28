# PCNN on the Xilinx ZCU102 (bare-metal, HLS accelerator)

Implements the single-zone **PCNN** from
[PCNN-Comparison](https://github.com/XuezhengWang/PCNN-Comparison) (`pcnn_epfl`,
EnergyPlus *Mid* building) as a custom FPGA accelerator on the ZCU102
(Zynq UltraScale+ XCZU9EG), with a bare-metal Vitis host application.

Model configuration (paper *Dual-phase evaluation framework...*, Table 6):
input MLP 2->32 (ReLU), LSTM 3 layers x 128 hidden (learned initial states),
LayerNorm, output MLP 128->32->1 (tanh), division factor 30, physics module
`E` with positive parameters `a`, `b`, `d` (interior zone, no neighbors).
One accelerator invocation = one 15-minute timestep; LSTM h/c states and the
`D`/`E` accumulators persist on-chip between invocations, exactly mirroring
the recurrent step-by-step use of the PyTorch model in `Model.predict()`.

## Layout

    python/pcnn_jax.py              JAX reimplementation: trains demo weights,
                                    exports headers + golden vectors
    python/export_weights_pytorch.py  exports a *real* repo checkpoint
                                    (best_model.pt) to the same headers
    harvard_to_pcnn.py              Converts Harvard SEC data to PCNN 'Mid' schema
                                    and generates HLS test vectors (tb_x/tb_y)
    hls/pcnn.h, pcnn.cpp            Vitis HLS top function `pcnn_step` (optimized PAR=16)
    hls/pcnn_tb.cpp                 C-sim testbench vs golden vectors (or ground truth tb_y.txt)
    hls/run_hls.tcl                 HLS build script (xczu9eg-ffvb1156-2-e)
    hls/weights/pcnn_weights.h      generated weights (~1.4 MB float, in BRAM)
    hls/weights/pcnn_config.h       generated dims/constants/normalization
    hls/tb_data/tb_x.txt            96-step test-day inputs (normalized)
    hls/tb_data/tb_ref.txt          golden T, D, E per step
    vitis/main.c                    bare-metal host app (Cortex-A53)

## Interface of the accelerator

    void pcnn_step(const float x[6], int mode,
                   float *T_out, float *D_out, float *E_out);

* `x` — normalized features `[T_out, solar, occ, T_room, P_hvac, case]`
  (min-max to [0.1, 0.9] with `RAW_MIN/RAW_MAX` from `pcnn_config.h`).
* `mode` — `0` reset state + warm-start step, `1` warm-start step
  (true room temperature in `x[3]`), `2` autoregressive prediction step.
* All ports on one AXI4-Lite `ctrl` bundle (block-level `ap_ctrl_hs`).

Protocol per horizon (same as the repo: `warm_start_length = 8`):
step 0 with mode 0, steps 1..7 with mode 1, then mode 2 for up to 88
prediction steps (24 h total at 15 min).

## Build flow

### 1. Weights
Demo weights are already generated. For real weights, train with the repo
(hyperparameters above), then from the repo root:

    python3 export_weights_pytorch.py --ckpt saves/models/.../best_model.pt \
        --csv "Processed E+ data/Mid.csv" --out <this dir>/hls/weights

`tb_x.txt`/`tb_ref.txt` are regenerated with the PyTorch model, so the HLS
C-sim re-verifies bit-consistency against PyTorch.

### 2. Harvard Dataset Testing (Optional)
You can generate test vectors from real-world data and test the accelerator accuracy natively using standard C++ (no Xilinx tools required):

    # 1. Generate test vectors
    ./harvard_to_pcnn.py --harvard "data_...csv" --mid "Mid.csv" \
        --config pcnn_zcu102/hls/weights/pcnn_config.h --outdir ./out --rooms 5.417

    # 2. Run C-Simulation using native GCC
    cd pcnn_zcu102/hls
    g++ -I./weights -I. pcnn.cpp pcnn_tb.cpp -o pcnn_sim
    ./pcnn_sim ../../out/5.417_tb_x.txt ../../out/5.417_tb_y.txt

The testbench dynamically detects the 1-column `tb_y.txt` and reports RMSE against the ground truth room temperature.

### 3. HLS IP

    cd hls
    vitis_hls -f run_hls.tcl

Runs C-sim (must print `TEST PASSED`), synthesis, and exports the IP to
`pcnn_hls/sol1/impl/ip`. Timing: 100 MHz default; ~0.4–1.5 ms/step depending
on the achieved II of the dot-product loops — vastly faster than the 15-min
control interval, and deterministic (no OS jitter).

### 4. Vivado block design (Vivado 2022.x+)
1. New project, board = ZCU102. Create block design.
2. Add **Zynq UltraScale+ MPSoC**, run block automation (apply board preset).
3. Settings: enable one PS-PL master (M_AXI_HPM0_FPD or LPD).
4. IP catalog -> add repository -> `hls/pcnn_hls/sol1/impl/ip` -> drop
   **Pcnn_step** into the design.
5. Run connection automation (AXI SmartConnect PS->IP, clock `pl_clk0`
   100 MHz, `pl_resetn0`). Assign address (default 4K window is fine).
6. Validate, create HDL wrapper, generate bitstream, **File -> Export ->
   Export Hardware (include bitstream)** -> `pcnn_zcu102.xsa`.

### 5. Vitis bare-metal app
1. `vitis -w work` -> platform project from `pcnn_zcu102.xsa`
   (standalone, psu_cortexa53_0).
2. Application project on that platform; add `vitis/main.c`, the generated
   `hls/weights/pcnn_config.h`, and a `demo_sequence.h` (generate with:
   `python/make_demo_header.py hls/tb_data > vitis/demo_sequence.h`).
3. The HLS-generated driver (`xpcnn_step.*`) is picked up from the exported
   IP automatically by the BSP.
4. Board: SW6 = JTAG boot, connect USB-UART (115200). Run/Debug from Vitis.
   Expected UART output: per-step predictions, µs/step timing, `PASS`.

## Verification chain

    JAX/PyTorch golden model --> tb_ref.txt --> HLS C-sim (pcnn_tb.cpp)
        --> (optional) cosim_design RTL vs C --> on-board check in main.c

All stages compare T, D, E with tolerance 1e-3 in normalized units
(~0.026 degC for this dataset).

## Notes & knobs
* **Float vs fixed-point**: `data_t` is `float` for exact parity with
  PyTorch. For a smaller/faster design, switch to `ap_fixed<24,8>` and
  re-run C-sim to measure quantization error (E-module constants span
  ~1e-3..1e2, so keep >= 16 fractional bits).
* **Throughput**: `PAR` in `pcnn.cpp` is optimized to 16 with `ARRAY_PARTITION`
  on the weight arrays for massive parallel MACs; the ZCU9EG
  has 2520 DSPs, the model needs ~345k MACs/step.
* **Padded sequences**: the training-time zeroing of padded outputs
  (`x[..,0] < 1e-6`) is a batching artifact and intentionally omitted.
* **Multi-zone (S-PCNN/M-PCNN)**: same skeleton — duplicate the E-module
  per zone and widen the output MLP; weights export scripts extend directly.
