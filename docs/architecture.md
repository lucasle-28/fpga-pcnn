# Architecture

## The model

PCNN splits room-temperature prediction into two parallel streams that are summed at every step.

```
                         ┌─────────────────────────────────────────────┐
  Solar radiation ──┐    │            D Module (data-driven)           │
  Occupancy ────────┼──▶ │  Input MLP(2→32, ReLU)                      │
                    │    │    → LSTM(3 layers × 128 hidden)            │──▶ D_k+1 = nn/30 + T̃_k
                    │    │    → LayerNorm(128)                         │
                    │    │    → Output MLP(128→32→1, tanh)             │
                    │    └─────────────────────────────────────────────┘
                    │                                                       T = D + E
  Outdoor temp ─────┤    ┌─────────────────────────────────────────────┐
  Room temp ────────┤    │            E Module (physics)               │
  HVAC power ───────┤    │  Heat loss:  E -= b·(T_room − T_outdoor)    │──▶ E_k+1
  Heating/cooling ──┘    │  HVAC gain:  E += a·u (heat) or d·u (cool)  │
                         └─────────────────────────────────────────────┘
```

The D module learns whatever the physics model cannot express (solar gain through glazing, occupant heat, thermal mass); the E module enforces energy-balance behavior that a pure network would violate when extrapolating. `a`, `b`, `d` are the learned physics coefficients `A_EFF`, `B_EFF`, `D_EFF` in `pcnn_config.h`.

One accelerator invocation = one 15-minute timestep. LSTM hidden/cell states and the D/E accumulators persist in on-chip BRAM between invocations, exactly mirroring the step-by-step recurrence of `Model.predict()` in the original PyTorch codebase.

A 24-hour horizon is **96 steps**:

- **Steps 0–7 — warm-start**: feed true measured room temperature to condition the LSTM state.
- **Steps 8–95 — autoregressive**: the model feeds its own prediction back, forecasting up to 22 h ahead.

`mode` selects which: `0` resets state then warm-starts, `1` warm-starts, `2` predicts autoregressively.

## Feature normalization

All six features are min-max scaled to `[0.1, 0.9]` using `RAW_MIN` / `RAW_MAX` from `pcnn_config.h`. The `[0.1, 0.9]` band (rather than `[0, 1]`) leaves headroom so real-world values slightly outside the training range do not clip.

To convert a normalized temperature back to °C:

```c
degC = (norm - 0.1f) / 0.8f * (RAW_MAX[COL_T] - RAW_MIN[COL_T]) + RAW_MIN[COL_T];
```

An RMSE in normalized units scales by `(RAW_MAX[COL_T] - RAW_MIN[COL_T]) / 0.8` ≈ **26.14 °C per unit**. Both `main.c` and `pcnn_tb.cpp` use this factor — keep them consistent if you retrain.

`case` (feature 5) is a ternary flag selecting the HVAC branch inside the E module; `pcnn_step` branches on `> 0.5`, so it must arrive as `0.1` / `0.5` / `0.9`, never raw `-1/0/1`.

## HLS design decisions

Each of these was a response to a specific synthesis failure — the reasoning matters more than the value.

- **Float32 throughout.** `data_t` is `float`, chosen for exact parity with PyTorch so any discrepancy is a logic bug rather than a quantization artifact. To shrink the design, switch to `ap_fixed<24,8>` and re-run C-sim to measure the error. Keep ≥ 16 fractional bits: the physics constants span ~1e-3 to ~1e+2 (`B_EFF` is 1.25e-04), so aggressive truncation destroys the E module first.

- **LUT-based sigmoid.** A 256-entry table over `[-8, 8]` with linear interpolation (`sigm_lut256.inc`) replaces the HLS `expf` polynomial core, which costs ~14 DSPs per instance. `tanhf` still uses the HLS built-in — `generic_tanh_float_s` is 5 DSPs, cheaper than a second table plus its interpolation logic. That asymmetry is deliberate, not an oversight.

- **PAR=4 dot-product lanes.** Started at 8; synthesis over-ran the ZCU102's 2,520 DSPs, so it was halved. Each lane keeps a partial sum in a fully partitioned array (`ARRAY_PARTITION complete` on `ps`) with `PIPELINE` on the inner loop, then the lanes reduce at the end.

- **Explicit DSP budget caps.** `#pragma HLS ALLOCATION operation instances=fmul limit=650` and `fadd limit=600` stop HLS from instantiating more floating-point operators than the device can place. Without these, the tool happily produces a design that fails implementation rather than backing off on its own.

- **State array partitioning.** `h` and `c` are partitioned `complete` on the layer dimension (only 3 layers) and `cyclic factor=8` on the hidden dimension (128) — enough banking to feed the PAR lanes without exploding into 128 separate registers per layer.

- **Weights entirely on-chip.** ~1.4 M float32 parameters live in BRAM/LUTROM, which is why BRAM (42%) is the binding resource. This buys deterministic latency with no DRAM traffic; the cost is that a larger model will not fit without off-chip streaming.

## Extending to multiple zones

The single-zone E module can be duplicated per zone and the output MLP widened to produce one temperature per zone (the S-PCNN / M-PCNN variants). The weight-export scripts already carry the zone dimension through, so the change is mostly in `pcnn.cpp` and the accelerator interface.
