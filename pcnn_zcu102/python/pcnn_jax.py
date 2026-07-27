#!/usr/bin/env python3
"""
PCNN (EPFL single-zone variant, config from Wang et al. 2026, Table 6) -- JAX
reimplementation used to train demo weights and generate the HLS golden
reference when PyTorch is unavailable.

Architecture (identical to pcnn_epfl/module.py :: PCNN):
  input MLP  : 2 -> 32, ReLU              (inputs_D = [solar, occupancy])
  LSTM       : 3 layers, hidden 128       (learned initial h/c states)
  layer norm : 128
  output MLP : 128 -> 32 (tanh) -> 1 (tanh)
  D_k+1 = nn_out / division_factor + T~_k          (division_factor = 30)
  E_k+1 = E_k - b*(T_k - T_out_k)/b_scal + [a*u/a_scal | d*u/d_scal]
  T = D + E

Usage:
  python3 pcnn_jax.py --csv Mid.csv --train-seconds 35   # repeat until loss ok
  python3 pcnn_jax.py --csv Mid.csv --export out_dir     # headers + golden
"""
import argparse, os, pickle, time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
jax.config.update('jax_compilation_cache_dir', '/tmp/jaxcache')
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)

INTERVAL = 15
WARM = 8
SEQ = 96
STRIDE = 4
IN_NN = 32
H = 128
L = 3
OUT_NN = 32
DIV = 30.0
LR = 5e-4
LR_PHY = LR * 10.0
BATCH = 128
SEED = 42

COL_OUT, COL_SOL, COL_OCC, COL_T, COL_P, COL_CASE = 0, 1, 2, 3, 4, 5
X_COLS = ['Outdoor air temperature [C]', 'Solar radiation [W/m2]',
          'Occupant count', 'Room temperature [C]', 'HVAC load [W]']

def load_dataset(csv):
    cache = '/tmp/pcnn_data.npz'
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z['X'], z['Y'], z['consts'].item()
    d = pd.read_csv(csv, index_col=0)
    df = d[X_COLS].copy()
    case = np.full(len(df), -1.0)
    case[d['HVAC load [W]'].values == 0] = 0.0
    case[d['HVAC load [W]'].values > 0] = 1.0
    df['Case'] = case
    mn, mx = df.min(), df.max()
    norm = 0.8 * (df - mn) / (mx - mn) + 0.1
    X = norm.values[:-1].astype(np.float32)
    Y = norm['Room temperature [C]'].values[1:].astype(np.float32)

    diffT = float(mx[X_COLS[3]] - mn[X_COLS[3]]); minT = float(mn[X_COLS[3]])
    diffO = float(mx[X_COLS[0]] - mn[X_COLS[0]]); minO = float(mn[X_COLS[0]])
    diffP = float(mx[X_COLS[4]] - mn[X_COLS[4]]); minP = float(mn[X_COLS[4]])
    zero_power = 0.8 * (0.0 - minP) / diffP + 0.1
    b_scal = 1.0 / (1.5 / diffT * 0.8 / 25 / 6 / 60 * INTERVAL)
    c_scal = b_scal / 10.0
    a_scal = 1.0 / (2.0 / diffT * 0.8 / (1000.0 / diffP * 0.8) / (4 * 60 / INTERVAL))
    d_scal = a_scal
    consts = dict(diffT=diffT, minT=minT, diffO=diffO, minO=minO,
                  diffP=diffP, minP=minP, zero_power=zero_power,
                  a_scal=a_scal, b_scal=b_scal, c_scal=c_scal, d_scal=d_scal,
                  mins=mn.values.astype(np.float64), maxs=mx.values.astype(np.float64),
                  cols=list(df.columns))
    np.savez(cache, X=X, Y=Y, consts=consts)
    np.savez(cache, X=X, Y=Y, consts=consts)
    return X, Y, consts

def xavier(rng, shape):
    fan_out, fan_in = shape
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, shape).astype(np.float32)

def init_params(rng):
    p = {}
    p['Win'] = xavier(rng, (IN_NN, 2)); p['bin'] = np.zeros(IN_NN, np.float32)
    for l in range(L):
        isz = IN_NN if l == 0 else H
        p[f'Wih{l}'] = xavier(rng, (4 * H, isz))
        p[f'Whh{l}'] = xavier(rng, (4 * H, H))
        p[f'bih{l}'] = np.zeros(4 * H, np.float32)
        p[f'bhh{l}'] = np.zeros(4 * H, np.float32)
    p['ln_g'] = np.ones(H, np.float32); p['ln_b'] = np.zeros(H, np.float32)
    p['W1'] = xavier(rng, (OUT_NN, H)); p['b1'] = np.zeros(OUT_NN, np.float32)
    p['W2'] = xavier(rng, (1, OUT_NN)); p['b2'] = np.zeros(1, np.float32)
    p['h0'] = np.zeros((L, H), np.float32); p['c0'] = np.zeros((L, H), np.float32)
    for k in ['a_log', 'b_log', 'c_log', 'd_log']:
        p[k] = np.zeros((), np.float32)
    return {k: jnp.asarray(v) for k, v in p.items()}

PHY_KEYS = ('a_log', 'b_log', 'c_log', 'd_log')

def make_forward(consts):
    zp = consts['zero_power']; diffT = consts['diffT']; minT = consts['minT']
    diffO = consts['diffO']; minO = consts['minO']
    a_s = consts['a_scal']; b_s = consts['b_scal']; d_s = consts['d_scal']

    def cell(p, l, inp, h, c):
        g = p[f'Wih{l}'] @ inp + p[f'bih{l}'] + p[f'Whh{l}'] @ h + p[f'bhh{l}']
        i, f, gg, o = jnp.split(g, 4)
        cn = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(gg)
        hn = jax.nn.sigmoid(o) * jnp.tanh(cn)
        return hn, cn

    def step(p, carry, xt, warm):
        h, c, lastD, lastE = carry
        x_temp = jnp.where(warm, xt[COL_T] - lastE, lastD)
        emb = jax.nn.relu(p['Win'] @ jnp.array([xt[COL_SOL], xt[COL_OCC]]) + p['bin'])
        inp = emb; hs = []; cs = []
        for l in range(L):
            hn, cn = cell(p, l, inp, h[l], c[l])
            hs.append(hn); cs.append(cn); inp = hn
        mu = jnp.mean(inp); var = jnp.mean((inp - mu) ** 2)
        ln = (inp - mu) / jnp.sqrt(var + 1e-5) * p['ln_g'] + p['ln_b']
        t1 = jnp.tanh(p['W1'] @ ln + p['b1'])
        t2 = jnp.tanh(p['W2'] @ t1 + p['b2'])[0]
        D = t2 / DIV + x_temp
        T_un = (x_temp + lastE - 0.1) / 0.8 * diffT + minT
        To_un = (xt[COL_OUT] - 0.1) / 0.8 * diffO + minO
        E = lastE - jnp.exp(p['b_log']) * (T_un - To_un) / b_s
        p_eff = jnp.where(xt[COL_P] > 0.05, xt[COL_P], zp)
        active = (jnp.abs(p_eff - zp) > 1e-6).astype(jnp.float32)
        pw = xt[COL_P] - zp
        heat = (xt[COL_CASE] > 0.5).astype(jnp.float32)
        cool = (xt[COL_CASE] < 0.5).astype(jnp.float32)
        E = E + active * (heat * jnp.exp(p['a_log']) * pw / a_s
                          + cool * jnp.exp(p['d_log']) * pw / d_s)
        T = D + E
        return (jnp.stack(hs), jnp.stack(cs), D, E), (T, D, E)

    def forward_seq(p, xseq):
        warm = jnp.arange(xseq.shape[0]) < WARM
        carry = (p['h0'], p['c0'], jnp.float32(0), jnp.float32(0))
        def body(carry, inp):
            xt, w = inp
            return step(p, carry, xt, w)
        _, (T, D, E) = jax.lax.scan(body, carry, (xseq, warm))
        return T, D, E

    return forward_seq

def carr(f, name, arr):
    arr = np.asarray(arr, np.float64).ravel()
    f.write(f'static const float {name}[{len(arr)}] = {{\n')
    for i in range(0, len(arr), 6):
        f.write('    ' + ', '.join(f'{x:.9e}f' for x in arr[i:i + 6]) + ',\n')
    f.write('};\n\n')

def export(outdir, params, consts, fwd, X, Y):
    os.makedirs(outdir, exist_ok=True)
    p = {k: np.asarray(v) for k, v in params.items()}

    with open(os.path.join(outdir, 'pcnn_weights.h'), 'w') as f:
        f.write('// Auto-generated by pcnn_jax.py -- PCNN weights (EP Mid building)\n')
        f.write('#ifndef PCNN_WEIGHTS_H\n#define PCNN_WEIGHTS_H\n\n')
        carr(f, 'W_IN', p['Win']);  carr(f, 'B_IN', p['bin'])
        for l in range(L):
            carr(f, f'LSTM_WIH{l}', p[f'Wih{l}'])
            carr(f, f'LSTM_WHH{l}', p[f'Whh{l}'])
            carr(f, f'LSTM_BIH{l}', p[f'bih{l}'])
            carr(f, f'LSTM_BHH{l}', p[f'bhh{l}'])
        carr(f, 'LN_G', p['ln_g']); carr(f, 'LN_B', p['ln_b'])
        carr(f, 'W_OUT1', p['W1']); carr(f, 'B_OUT1', p['b1'])
        carr(f, 'W_OUT2', p['W2']); carr(f, 'B_OUT2', p['b2'])
        carr(f, 'INIT_H', p['h0']); carr(f, 'INIT_C', p['c0'])
        f.write('#endif\n')

    a_eff = float(np.exp(p['a_log']) / consts['a_scal'])
    b_eff = float(np.exp(p['b_log']) / consts['b_scal'])
    d_eff = float(np.exp(p['d_log']) / consts['d_scal'])
    with open(os.path.join(outdir, 'pcnn_config.h'), 'w') as f:
        f.write('// Auto-generated by pcnn_jax.py -- PCNN dimensions & constants\n')
        f.write('#ifndef PCNN_CONFIG_H\n#define PCNN_CONFIG_H\n\n')
        f.write(f'#define N_FEAT   6\n#define IN_NN    {IN_NN}\n')
        f.write(f'#define LSTM_H   {H}\n#define LSTM_L   {L}\n')
        f.write(f'#define OUT_NN   {OUT_NN}\n\n')
        f.write(f'#define COL_OUT  {COL_OUT}\n#define COL_SOL  {COL_SOL}\n')
        f.write(f'#define COL_OCC  {COL_OCC}\n#define COL_T    {COL_T}\n')
        f.write(f'#define COL_P    {COL_P}\n#define COL_CASE {COL_CASE}\n\n')
        f.write(f'#define DIV_FACTOR   {DIV}f\n')
        f.write(f'#define ZERO_POWER   {consts["zero_power"]:.9e}f\n')
        f.write(f'#define ROOM_MIN     {consts["minT"]:.9e}f\n')
        f.write(f'#define ROOM_DIFF    {consts["diffT"]:.9e}f\n')
        f.write(f'#define OUT_MIN      {consts["minO"]:.9e}f\n')
        f.write(f'#define OUT_DIFF     {consts["diffO"]:.9e}f\n')
        f.write(f'#define A_EFF        {a_eff:.9e}f\n')
        f.write(f'#define B_EFF        {b_eff:.9e}f\n')
        f.write(f'#define D_EFF        {d_eff:.9e}f\n\n')
        f.write('// raw-data min/max per column, for host-side normalization\n')
        for i, c in enumerate(consts['cols']):
            f.write(f'// col {i}: {c}\n')
        f.write('static const float RAW_MIN[6] = { ' +
                ', '.join(f'{x:.9e}f' for x in consts['mins']) + ' };\n')
        f.write('static const float RAW_MAX[6] = { ' +
                ', '.join(f'{x:.9e}f' for x in consts['maxs']) + ' };\n')
        f.write('#endif\n')

    s = 211 * 96
    xseq = X[s:s + SEQ]
    T, D, E = fwd(params, jnp.asarray(xseq))
    np.savetxt(os.path.join(outdir, 'tb_x.txt'), xseq, fmt='%.9e')
    ref = np.stack([np.asarray(T), np.asarray(D), np.asarray(E)], axis=1)
    np.savetxt(os.path.join(outdir, 'tb_ref.txt'), ref, fmt='%.9e')
    np.savetxt(os.path.join(outdir, 'tb_y.txt'), Y[s:s + SEQ], fmt='%.9e')
    print('export complete ->', outdir)
    print('physics params a,b,c,d (log):',
          [float(p[k]) for k in PHY_KEYS])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--ckpt', default='/tmp/pcnn_ckpt.pkl')
    ap.add_argument('--train-seconds', type=float, default=0)
    ap.add_argument('--export', default=None)
    args = ap.parse_args()

    X, Y, consts = load_dataset(args.csv)
    fwd = make_forward(consts)
    train_end = 180 * 96
    starts = np.arange(0, train_end - SEQ, STRIDE)
    val_starts = np.arange(train_end, (210 * 96) - SEQ, SEQ)

    def loss_fn(p, xb, yb):
        T, _, _ = jax.vmap(lambda xs: fwd(p, xs))(xb)
        return jnp.mean((T - yb) ** 2)

    if os.path.exists(args.ckpt):
        with open(args.ckpt, 'rb') as f:
            st = pickle.load(f)
        params = {k: jnp.asarray(x) for k, x in st['params'].items()}
        m = {k: jnp.asarray(x) for k, x in st['m'].items()}
        v = {k: jnp.asarray(x) for k, x in st['v'].items()}
        it = st['it']; rng_state = st['rng']
        print(f'resumed at iter {it}')
    else:
        rng = np.random.default_rng(SEED)
        params = init_params(rng)
        m = jax.tree.map(jnp.zeros_like, params)
        v = jax.tree.map(jnp.zeros_like, params)
        it = 0; rng_state = np.random.default_rng(SEED + 1)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def adam_update(params, m, v, grads, t):
        b1, b2, eps = 0.9, 0.999, 1e-8
        out_p, out_m, out_v = {}, {}, {}
        for k in params:
            lr = LR_PHY if k in PHY_KEYS else LR
            g = grads[k]
            mk = b1 * m[k] + (1 - b1) * g
            vk = b2 * v[k] + (1 - b2) * g * g
            mh = mk / (1 - b1 ** t); vh = vk / (1 - b2 ** t)
            out_p[k] = params[k] - lr * mh / (jnp.sqrt(vh) + eps)
            out_m[k] = mk; out_v[k] = vk
        return out_p, out_m, out_v

    if args.train_seconds > 0:
        t0 = time.time()
        while time.time() - t0 < args.train_seconds:
            idx = rng_state.choice(len(starts), BATCH, replace=False)
            s = starts[idx]
            xb = np.stack([X[i:i + SEQ] for i in s])
            yb = np.stack([Y[i:i + SEQ] for i in s])
            it += 1
            lv, grads = grad_fn(params, jnp.asarray(xb), jnp.asarray(yb))
            params, m, v = adam_update(params, m, v, grads, jnp.float32(it))
            print(f'iter {it}  loss {float(lv):.6f}', flush=True)
            with open(args.ckpt + '.tmp', 'wb') as f:
                pickle.dump(dict(params={k: np.asarray(x) for k, x in params.items()},
                                 m={k: np.asarray(x) for k, x in m.items()},
                                 v={k: np.asarray(x) for k, x in v.items()},
                                 it=it, rng=rng_state), f)
            os.replace(args.ckpt + '.tmp', args.ckpt)
        print(f'saved checkpoint at iter {it}')

    if args.export:
        export(args.export, params, consts, fwd, X, Y)

if __name__ == '__main__':
    main()
