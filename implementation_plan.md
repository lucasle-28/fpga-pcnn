# Handoff Preparation for fpga-pcnn

Prepare the repo so another student can clone it, understand it, and run it with minimal friction.

## Current State

**Good news** — this project is already in very strong shape:
- README is thorough (246 lines, architecture diagrams, build flow, results)
- Code is well-commented with no stale TODOs/FIXMEs
- All source files are committed and pushed to `origin/main`
- Git history is clean (15 commits, sensible progression)

**Gaps I found:**

| Issue | Impact |
|-------|--------|
| No `requirements.txt` | Next student won't know they need `numpy` + `pandas` for the Python scripts |
| `.venv/` is 160 MB on disk (not in git, but not in `.gitignore` either) | Already gitignored ✓ — just confirming |
| 1 uncommitted change in `make_harvard_header.py` | A stray usage example pasted into the docstring — should be committed or reverted |
| Large generated files tracked in git (`pcnn_weights.h` 6.5 MB, `demo_sequence.h` 2.4 MB, `data_*.csv` 12 MB, `Mid.csv` 2.9 MB, `out/` 8.6 MB) | Makes the clone heavy (~30 MB). Fine for this project, but worth noting in a handoff doc |
| No `HANDOFF.md` or quickstart checklist | The README covers everything but is dense — a "first 10 minutes" guide would help |
| `README.md` mentions `sigm_lut256.inc` but weights/ also has `tanh_lut256.inc` (undocumented) | Minor — the file exists but isn't referenced in the project layout tree |
| Commit messages like `"oops"` and `"cleanup"` at HEAD | Cosmetic, not worth rewriting history |

## Proposed Changes

### 1. Add `requirements.txt`

#### [NEW] [requirements.txt](file:///home/lucas/Documents/pcnn/requirements.txt)

Pin the two Python dependencies the scripts actually import:

```
numpy>=1.24
pandas>=2.0
```

The JAX script (`pcnn_jax.py`) has its own heavier deps (jax, jaxlib, torch for export) — add a note but don't pin them since that path is optional.

---

### 2. Update `.gitignore`

#### [MODIFY] [.gitignore](file:///home/lucas/Documents/pcnn/.gitignore)

Add common HLS/Vivado build artifacts so the next student doesn't accidentally commit synthesis output:

```diff
+# HLS synthesis output
+pcnn_zcu102/hls/pcnn_hls/
+
+# Vivado project output
+*.xpr
+*.jou
+*.log
+vivado_*/
+.Xil/
+
+# Generated output directory (regenerate with harvard_to_pcnn.py)
+# Uncomment to exclude: out/
```

---

### 3. Commit the stray change in `make_harvard_header.py`

#### [MODIFY] [make_harvard_header.py](file:///home/lucas/Documents/pcnn/pcnn_zcu102/python/make_harvard_header.py)

The uncommitted diff adds a second usage example to the docstring. I'll clean it up to be a proper part of the usage block rather than a stray paste.

---

### 4. Fix the project layout tree in README

#### [MODIFY] [README.md](file:///home/lucas/Documents/pcnn/README.md)

- Add the missing `tanh_lut256.inc` to the tree
- Add `requirements.txt` to the tree
- Minor: add a note about `photos/` contents

---

### 5. Add `HANDOFF.md` — quickstart for the next student

#### [NEW] [HANDOFF.md](file:///home/lucas/Documents/pcnn/HANDOFF.md)

A concise "first 10 minutes" guide covering:
- What this project is (1 paragraph)
- Prerequisites (Vitis HLS 2022+, GCC, Python 3.10+, ZCU102 board)
- Quick-check: compile and run C-sim with just GCC (no Xilinx tools)
- Where to go next (pointers into README sections)
- Known gotchas / things the next student should know
- Contact info placeholder

---

## Open Questions

> [!IMPORTANT]
> **Large files in git**: The repo tracks ~30 MB of generated/data files (`pcnn_weights.h`, `demo_sequence.h`, CSVs, `out/`). These are needed for the next student to immediately run tests without regenerating. Do you want to keep them tracked, or move them to Git LFS / a download script?

> [!NOTE]
> **Contact info**: Should the HANDOFF.md include your name/email, or should I leave a placeholder?

## Verification Plan

### Automated Tests
```bash
# Confirm C-sim still compiles and runs
cd pcnn_zcu102/hls && g++ -I./weights -I. pcnn.cpp pcnn_tb.cpp -o pcnn_sim && ./pcnn_sim
```

### Manual Verification
- Review the HANDOFF.md for clarity
- Verify `.gitignore` additions don't exclude anything currently tracked
- `git status` should be clean after the final commit
