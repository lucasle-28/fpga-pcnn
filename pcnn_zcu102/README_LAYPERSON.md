# Building Temperature Predictor — Running on Custom Hardware

## What This Project Does

This project takes a **machine-learning model that predicts building temperatures** and runs it directly on a specialized hardware chip (an FPGA), rather than on a regular computer.

### The Problem It Solves

Modern buildings use "smart" heating and cooling systems that try to predict what the indoor temperature *will be* in the near future, so they can adjust the HVAC (heating, ventilation, and air conditioning) proactively — not just react after a room is already too hot or cold.

The prediction model used here is called a **PCNN** (Physics-Constrained Neural Network). It's a type of neural network that doesn't just learn patterns from data — it also respects real physical laws about how buildings heat up and cool down. This makes its predictions more trustworthy and physically plausible than a purely data-driven model.

### Why Run It on an FPGA?

An FPGA (Field-Programmable Gate Array) is a chip that can be rewired to perform specific computations very efficiently. Compared to running the model on a regular CPU or even a GPU:

- **Speed**: The chip produces a prediction in roughly 0.4–1.5 milliseconds — far faster than the 15-minute intervals between each control decision.
- **Determinism**: Unlike a computer running an operating system, the FPGA always takes the same amount of time. There are no background tasks or software interruptions that could introduce timing jitter. This predictability is important for safety-critical control systems.
- **Standalone operation**: The system runs "bare-metal" — no operating system, no Python runtime, no dependencies. Once programmed, the chip just runs.

The specific board used is a **Xilinx ZCU102**, a development board featuring a powerful Zynq UltraScale+ chip that combines a regular processor (ARM Cortex-A53) with programmable FPGA fabric on the same die.

---

## How the Model Works (Simplified)

The PCNN takes in **6 inputs** at each time step (every 15 minutes):

| Input | What It Represents |
|---|---|
| `T_out` | Outdoor temperature |
| `solar` | Solar radiation hitting the building |
| `occ` | Occupancy (how many people are inside) |
| `T_room` | Current room temperature |
| `P_hvac` | HVAC power being used |
| `case` | A scenario identifier |

From these inputs, the model produces **3 outputs**:

| Output | What It Represents |
|---|---|
| **T** | Predicted room temperature |
| **D** | A "data-driven" component — what the neural network learned from patterns in the data |
| **E** | A "physics" component — what the physical equations say should happen |

The key idea: **T = D + E**. The final prediction blends a learned component with a physics-based one. This is the "physics-constrained" part — the model can't just make up any answer; it must be consistent with thermal physics.

### Inside the Model

Under the hood, the model processes each time step through several stages:

1. **Input preparation** — A small neural network layer expands the 6 raw inputs into a richer 32-number representation.
2. **Memory (LSTM)** — A recurrent neural network layer that maintains a "memory" of recent conditions. This is what allows the model to understand trends (e.g., "the building has been warming up for the last hour"). It uses 3 stacked layers, each with 128 memory cells.
3. **Normalization** — A layer that stabilizes the numbers to keep training efficient.
4. **Output** — Another small neural network that compresses the 128 memory outputs down to a single number: the data-driven prediction **D**.
5. **Physics module** — A separate calculation using physical parameters (`a`, `b`, `d`) that computes **E** based on the energy balance of the building zone.
6. **Combination** — T = D + E, giving the final temperature prediction.

The LSTM's memory (called "hidden state" and "cell state") and the physics accumulators persist between time steps, so the model truly runs step-by-step, just like it does in the original Python software.

---

## What's in Each Folder

| Folder / File | Purpose |
|---|---|
| `python/` | Python scripts for training, exporting model weights, and generating test data |
| `hls/` | The hardware design — C++ code that gets compiled into FPGA logic |
| `hls/weights/` | The trained model parameters, stored as numbers in header files |
| `hls/tb_data/` | Test data: 96 time steps (one day at 15-min intervals) of inputs and expected outputs |
| `vitis/` | The software that runs on the ARM processor and talks to the FPGA accelerator |

### Key Files

- **`python/pcnn_jax.py`** — A reimplementation of the model in JAX (a numerical computing library). Used to generate demo weights and "golden" reference outputs for testing.
- **`python/export_weights_pytorch.py`** — Exports weights from a real, trained PyTorch model into a format the hardware can use.
- **`hls/pcnn.cpp` / `pcnn.h`** — The core hardware design. This C++ code describes the computation that will be "burned" into the FPGA fabric.
- **`hls/pcnn_tb.cpp`** — A test file that verifies the hardware design produces the same answers as the Python model.
- **`vitis/main.c`** — The program that runs on the ARM processor, feeds data to the FPGA accelerator, and reads back predictions.

---

## How to Build and Run It

### Step 1: Prepare the Model Weights

The project comes with demo weights already included. If you have your own trained model (a `best_model.pt` file from PyTorch training), you can export it:

```
python3 export_weights_pytorch.py --ckpt path/to/best_model.pt \
    --csv "Processed E+ data/Mid.csv" --out hls/weights
```

This converts the trained neural network parameters into C header files that the hardware can read.

### Step 2: Build the Hardware Design

```
cd hls
vitis_hls -f run_hls.tcl
```

This step:
1. **Simulates** the design in software, checking that it matches the Python model's outputs (must print `TEST PASSED`).
2. **Synthesizes** the design — translates the C++ description into actual FPGA logic gates, memory blocks, and arithmetic units.
3. **Exports** the result as a reusable "IP block" (Intellectual Property block — a packaged hardware module).

### Step 3: Create the Full Chip Design (Vivado)

Using Xilinx Vivado (the FPGA design tool):

1. Create a new project targeting the ZCU102 board.
2. Add the main processor (Zynq UltraScale+ MPSoC) and apply the board's default settings.
3. Drop in the PCNN accelerator block from Step 2.
4. Connect them together — the processor talks to the accelerator over an AXI bus (a standard on-chip communication protocol).
5. Generate the bitstream — the binary file that configures the FPGA.
6. Export the complete hardware specification (`.xsa` file).

### Step 4: Build and Run the Software

Using Xilinx Vitis (the software development tool):

1. Create a platform project from the `.xsa` file.
2. Create an application project and add the host program (`main.c`) and configuration headers.
3. Connect the board via USB, set it to JTAG boot mode, and run.

The program will output predicted temperatures for each time step, how long each prediction took (in microseconds), and whether the results match the expected values (`PASS`).

---

## How It's Verified

The project uses a multi-stage verification chain to ensure nothing goes wrong at any step:

```
Python model (JAX / PyTorch)
    ↓  generates expected outputs
HLS C simulation (software simulation of the hardware)
    ↓  compared against expected outputs
(Optional) RTL co-simulation (cycle-accurate hardware simulation)
    ↓  compared against C simulation
On-board execution (actual FPGA running on the ZCU102)
    ↓  compared against expected outputs
```

At every stage, the predicted temperature (T), data-driven component (D), and physics component (E) must match within a tolerance of **0.001 normalized units** — which corresponds to about **0.026 °C** for this particular dataset. That's a tiny margin of error.

---

## Design Choices and Tuning Options

### Precision: Floating-Point vs. Fixed-Point

Currently, the design uses standard 32-bit floating-point numbers — the same format as the Python model — to ensure the hardware produces *exactly the same results*. This uses more FPGA resources but eliminates any concern about numerical differences.

For a smaller and faster design, the numbers could be represented in **fixed-point** format (a simpler representation where the decimal point is in a fixed position). This would use fewer resources but introduces small rounding differences that need to be carefully checked.

### Throughput: Going Faster

The design can be made faster by computing more multiply-and-add operations in parallel. The FPGA has 2,520 DSP (Digital Signal Processing) blocks available, and the model needs about 345,000 multiply-accumulate operations per time step. Increasing parallelism trades more chip area for faster execution.

### Multi-Zone Buildings

The current model handles a single building zone (one thermal area). For buildings with multiple zones (e.g., different floors or wings), the architecture extends naturally: duplicate the physics module for each zone and widen the output layer. The weight export scripts are designed to support this.

---

## Quick Reference

| Term | What It Means |
|---|---|
| **PCNN** | Physics-Constrained Neural Network — a model that combines learned patterns with physical equations |
| **FPGA** | Field-Programmable Gate Array — a chip whose logic circuits can be reconfigured |
| **ZCU102** | A Xilinx development board with both a regular processor and an FPGA |
| **HLS** | High-Level Synthesis — compiling C/C++ code into hardware circuits |
| **LSTM** | Long Short-Term Memory — a type of neural network that can remember information over time |
| **Bare-metal** | Running directly on hardware without an operating system |
| **Bitstream** | The binary file that programs/configures an FPGA |
| **AXI** | A standard protocol for components on a chip to communicate with each other |
| **DSP block** | A dedicated arithmetic unit on the FPGA, optimized for multiply-and-add operations |
| **Warm-start** | Feeding the model real measured data for several steps before asking it to predict on its own |
