# Build Flow

Four stages. Only stage 2 needs no Xilinx tools; stages 3–5 need Vitis HLS / Vivado / Vitis **2022.x or newer** (developed on 2025.1).

## 1. Weight export

**Option A — real PyTorch checkpoint** (what the committed weights came from):

```bash
# run from the PCNN-Comparison repo root
python3 pcnn_zcu102/python/export_weights_pytorch.py \
    --ckpt saves/models/.../best_model.pt \
    --csv "Processed E+ data/Mid.csv" \
    --out pcnn_zcu102/hls/weights
```

**Option B — JAX demo weights** (no PyTorch, no checkpoint needed):

```bash
python3 pcnn_zcu102/python/pcnn_jax.py --csv Mid.csv --train-seconds 120
python3 pcnn_zcu102/python/pcnn_jax.py --csv Mid.csv --export pcnn_zcu102/hls/weights
```

Either path writes `pcnn_weights.h`, `pcnn_config.h`, `sigm_lut256.inc`, and golden vectors `tb_data/tb_x.txt` + `tb_ref.txt`.

> After regenerating: copy `hls/weights/pcnn_config.h` over `vitis/pcnn_config.h`, and regenerate `vitis/demo_sequence.h` (step 5). Stale copies fail silently — the numbers just get worse.

## 2. Test against real data with plain GCC

```bash
# Harvard BMS export -> normalized test vectors
./harvard_to_pcnn.py --harvard "data_2024-04-15_00-00-00_to_2025-06-20_00-00-00.csv" \
    --mid "Mid.csv" --config pcnn_zcu102/hls/weights/pcnn_config.h \
    --outdir ./out --rooms 5.417

cd pcnn_zcu102/hls
g++ -O2 -I./weights -I. pcnn.cpp pcnn_tb.cpp -o pcnn_sim
./pcnn_sim                                                     # golden vectors
./pcnn_sim ../../out/5.417_tb_x.txt ../../out/5.417_tb_y.txt   # real data
```

The testbench auto-detects its reference file: 3 columns (`tb_ref.txt`) → compare T/D/E against the golden model and pass/fail at 1e-3; 1 column (`tb_y.txt`) → report RMSE against ground truth. With no arguments it walks up to 6 directories looking for `tb_data/`, which is how Vitis HLS invokes it from its nested build directory.

`harvard_to_pcnn.py` also writes `<room>_segments.csv` and `<room>_mid_schema.csv` as intermediates for inspection; they are not tracked and nothing downstream reads them.

## 3. HLS synthesis

```bash
cd pcnn_zcu102/hls
vitis_hls -f run_hls.tcl
```

Runs C-simulation (**must** print `TEST PASSED` — synthesis is pointless otherwise), synthesizes at 100 MHz for `xczu9eg-ffvb1156-2-e`, and exports IP to `pcnn_hls/sol1/impl/ip`. Expect a long run; the design is large.

RTL/C co-simulation (`cosim_design`) is commented out — it is extremely slow with this much persistent state. Uncomment it if you change the datapath and want RTL-level confirmation before going to the board.

## 4. Vivado block design

1. New project targeting the **ZCU102** board.
2. Create block design → add **Zynq UltraScale+ MPSoC** → run block automation.
3. Enable one PS-PL AXI master port (`M_AXI_HPM0_FPD` or LPD).
4. Add IP repository → `hls/pcnn_hls/sol1/impl/ip` → drop in **Pcnn_step**.
5. Run connection automation (SmartConnect, `pl_clk0` at 100 MHz, `pl_resetn0`).
6. Validate → create HDL wrapper → generate bitstream → **Export Hardware** *(include bitstream)* → `pcnn_zcu102.xsa`.

See `photos/block-design.png` for the finished design.

## 5. Bare-metal application

1. Create a platform project from `pcnn_zcu102.xsa` (standalone, `psu_cortexa53_0`).
2. Create an application project; add `vitis/main.c`, `vitis/pcnn_config.h`, and `vitis/demo_sequence.h`.
3. Generate the embedded test data first:

   ```bash
   cd pcnn_zcu102/python
   python3 make_harvard_header.py ../../out/5.417_tb_x.txt ../../out/5.417_tb_y.txt \
       --horizons 195 > ../vitis/demo_sequence.h
   ```

   195 horizons is ~2.4 MB of header. Use `--horizons 1` while bringing the board up — it builds far faster.
4. The HLS-generated driver (`xpcnn_step.*`) is pulled in automatically by the BSP.
5. Board setup: **SW6 → JTAG boot**, USB-UART at **115200 baud**.
6. Run/Debug from Vitis. The UART prints per-step predictions in °C, per-horizon RMSE, and an overall figure.

## Verification chain

```
JAX/PyTorch golden model ──▶ tb_ref.txt ──▶ HLS C-sim (pcnn_tb.cpp)
      ──▶ (optional) cosim_design RTL vs C ──▶ on-board run in main.c
```

Golden comparisons use a 1e-3 tolerance in normalized units (~0.026 °C). The Harvard path has no golden model, so it reports ground-truth RMSE instead. Measured results: [results.md](results.md).
