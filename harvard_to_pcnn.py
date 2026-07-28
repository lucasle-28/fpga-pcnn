#!/usr/bin/env python3
"""
Harvard SEC processed data  ->  PCNN-Comparison 'Mid.csv' schema
                            ->  normalized test vectors for fpga-pcnn

    python harvard_to_pcnn.py \
        --harvard "Processed Data/data_2024-04-15_..._to_2025-06-20_....csv" \
        --mid     "Processed E+ data/Mid.csv" \
        --config  hls/weights/pcnn_config.h \
        --outdir  ./out --rooms 5.417 5.416

Per room:
    <room>_mid_schema.csv   Mid.csv layout, Celsius, signed watts
    <room>_tb_x.txt         normalized x[6], one row per step  (drop-in for HLS tb)
    <room>_tb_y.txt         normalized next-step room temp (ground truth)
    <room>_segments.csv     contiguous gap-free runs >= 96 steps
    norm_constants.json     RAW_MIN / RAW_MAX actually used
"""

import argparse, json, pathlib, re
import numpy as np
import pandas as pd

MID_COLS = ["DateTime", "Outdoor air temperature [C]", "Solar radiation [W/m2]",
            "Occupant count", "Room temperature [C]", "Heating setpoint [C]",
            "Cooling setpoint [C]", "HVAC load [W]"]

# pcnn_config.h column order: COL_OUT COL_SOL COL_OCC COL_T COL_P COL_CASE
FEATURES = ["T_out", "solar", "occ", "T_room", "P_hvac", "case"]
NORM_LO, NORM_HI = 0.1, 0.9
HORIZON, WARM_START = 96, 8


def f2c(f):
    """Harvard BMS logs °F; Mid.csv is °C."""
    return (f - 32.0) * 5.0 / 9.0


def derive_case(load):
    """
    x[5], per consts_from_csv() in export_weights_pytorch.py:
        load < 0 -> -1.0 cooling | == 0 -> 0.0 off | > 0 -> +1.0 heating
    Normalized like every other column (RAW_MIN[5]=-1, RAW_MAX[5]=+1) it becomes
    0.1 / 0.5 / 0.9 -- the values pcnn_step() branches on (>0.5 A_EFF,
    <0.5 D_EFF, ==0.5 neither, consistent because `active` is false at 0 W).
    """
    load = np.asarray(load, dtype=float)
    case = np.full(len(load), -1.0)
    case[load == 0] = 0.0
    case[load > 0] = 1.0
    return case


def parse_config_h(path):
    """RAW_MIN/RAW_MAX straight out of pcnn_config.h -- authoritative."""
    txt = pathlib.Path(path).read_text()
    out = {}
    for name in ("RAW_MIN", "RAW_MAX"):
        m = re.search(name + r"\s*\[6\]\s*=\s*\{([^}]*)\}", txt)
        if not m:
            raise ValueError(name + " not found in " + str(path))
        vals = [float(v.strip().rstrip("f")) for v in m.group(1).split(",") if v.strip()]
        out[name] = dict(zip(FEATURES, vals))
    return out["RAW_MIN"], out["RAW_MAX"]


def build_room_frame(h, room):
    load = h[f"{room} rcp load"] + h[f"{room} air load"]
    nan = pd.Series(np.nan, index=h.index)
    heat_sp = f2c(h[f"{room} Heating SP"]) if f"{room} Heating SP" in h else nan
    cool_sp = f2c(h[f"{room} Cooling SP"]) if f"{room} Cooling SP" in h else nan
    # Harvard uses 0 °F as an unoccupied/invalid setpoint sentinel (-17.8 °C).
    heat_sp = heat_sp.mask(heat_sp < 0.0)
    cool_sp = cool_sp.mask(cool_sp < 0.0)
    return pd.DataFrame({
        "DateTime": h.index,
        "Outdoor air temperature [C]": f2c(h["Thermometer"]).values,
        "Solar radiation [W/m2]": h["Solar Radiation Sensor"].values,
        "Occupant count": h[f"{room} occ"].values,     # remapped in main()
        "Room temperature [C]": f2c(h[f"{room} room temp"]).values,
        "Heating setpoint [C]": heat_sp.values,
        "Cooling setpoint [C]": cool_sp.values,
        "HVAC load [W]": load.values,
    })


def remap_occupancy(binary_occ, occ_lo, occ_hi):
    """
    Harvard `occ` is a binary flag; Mid.csv 'Occupant count' is a headcount --
    but its schedule is effectively two-state (the modal values 0.72567 and
    3.0 are ~71% of the training set, and are exactly RAW_MIN[2]/RAW_MAX[2]),
    so the flag maps onto 0.100 / 0.900 with no extrapolation.
    """
    return np.where(binary_occ > 0, occ_hi, occ_lo)


def find_segments(valid, min_len=HORIZON):
    v = np.asarray(valid, dtype=bool)
    if not v.any():
        return []
    e = np.flatnonzero(np.diff(np.concatenate(([False], v, [False]))))
    return [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a >= min_len]


def normalize(feats, raw_min, raw_max):
    normed, report = pd.DataFrame(index=feats.index), {}
    for c in FEATURES:
        lo, hi = raw_min[c], raw_max[c]
        n = NORM_LO + (NORM_HI - NORM_LO) * (feats[c] - lo) / (hi - lo)
        below, above = int((n < NORM_LO).sum()), int((n > NORM_HI).sum())
        report[c] = {"lo": below, "hi": above,
                     "pct": round(100.0 * (below + above) / max(len(n), 1), 3)}
        normed[c] = n.clip(NORM_LO, NORM_HI)
    return normed, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvard", required=True)
    ap.add_argument("--mid", required=True)
    ap.add_argument("--config", help="hls/weights/pcnn_config.h")
    ap.add_argument("--outdir", default="./out")
    ap.add_argument("--rooms", nargs="+", default=["5.417"])
    ap.add_argument("--longest-only", action="store_true")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(args.harvard, index_col=0, parse_dates=True).asfreq("15min")
    mid = pd.read_csv(args.mid, parse_dates=["DateTime"])

    if args.config:
        raw_min, raw_max = parse_config_h(args.config)
        print("RAW_MIN/RAW_MAX from", args.config)
    else:
        mf = pd.DataFrame({
            "T_out": mid["Outdoor air temperature [C]"],
            "solar": mid["Solar radiation [W/m2]"],
            "occ": mid["Occupant count"],
            "T_room": mid["Room temperature [C]"],
            "P_hvac": mid["HVAC load [W]"],
            "case": derive_case(mid["HVAC load [W]"].values)})
        raw_min, raw_max = mf.min().to_dict(), mf.max().to_dict()
        print("RAW_MIN/RAW_MAX derived from Mid.csv (pass --config to use the header)")

    (outdir / "norm_constants.json").write_text(
        json.dumps({"RAW_MIN": raw_min, "RAW_MAX": raw_max}, indent=2))

    for room in args.rooms:
        df = build_room_frame(h, room)
        df["Occupant count"] = remap_occupancy(
            df["Occupant count"].values, raw_min["occ"], raw_max["occ"])

        feats = pd.DataFrame({
            "T_out": df["Outdoor air temperature [C]"],
            "solar": df["Solar radiation [W/m2]"],
            "occ": df["Occupant count"],
            "T_room": df["Room temperature [C]"],
            "P_hvac": df["HVAC load [W]"],
            "case": derive_case(df["HVAC load [W]"].values)})

        valid = feats[FEATURES[:-1]].notna().all(axis=1).values
        segments = find_segments(valid)
        if not segments:
            print(f"[{room}] no gap-free run >= {HORIZON} steps -- skipped"); continue
        if args.longest_only:
            segments = [max(segments, key=lambda s: s[1] - s[0])]

        keep = np.zeros(len(df), dtype=bool)
        for a, b in segments:
            keep[a: a + ((b - a) // HORIZON) * HORIZON] = True

        df_keep = df[keep].reset_index(drop=True)
        feats_keep = feats[keep].reset_index(drop=True)
        df_keep[MID_COLS].to_csv(outdir / f"{room}_mid_schema.csv", index=False)

        normed, rep = normalize(feats_keep, raw_min, raw_max)
        # mirror export_weights_pytorch.py: X = norm[:-1], Y = norm(T_room)[1:]
        np.savetxt(outdir / f"{room}_tb_x.txt", normed[FEATURES].values[:-1], fmt="%.9e")
        np.savetxt(outdir / f"{room}_tb_y.txt", normed["T_room"].values[1:], fmt="%.9e")

        pd.DataFrame([{"start_idx": a, "stop_idx": b, "n_steps": b - a,
                       "n_horizons": (b - a) // HORIZON,
                       "start_time": h.index[a], "stop_time": h.index[b - 1]}
                      for a, b in segments]).to_csv(
            outdir / f"{room}_segments.csv", index=False)

        n_h = len(df_keep) // HORIZON
        print(f"[{room}] {len(df_keep):6d} steps  {n_h:4d} horizons "
              f"from {len(segments)} segment(s)")
        for c, r in rep.items():
            if r["pct"] > 0:
                print(f"         clipped {c:7s} {r['pct']:.3f}% (lo {r['lo']}, hi {r['hi']})")


if __name__ == "__main__":
    main()
