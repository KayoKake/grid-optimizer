
import sys
import os
import math
import random
import hashlib
from collections import namedtuple, Counter

Config = namedtuple('Config', 'objective bits_mult max_level uncap_pwr w h')

TARGET_BITS = 78.0

LVL_OUT     = 1.20
LVL_WEIGHT  = 2.0
LVL_COST_SCALE = 1.30
PWR_ABS_MAX = 20
SCALE_CONS_WITH_LEVEL = True

# Ledger keys pack each cell's level into 5 bits (see _pack_grid), so no tier may
# exceed 31. If PWR_ABS_MAX or the GUI's max-tier cap is ever raised past this,
# widen the packing first.
assert PWR_ABS_MAX <= 31, "level must fit in 5 bits for ledger keys"

OVR_BASE    = 1.00
OVR_PER_TIER = 0.20

def ovr_strength(lvl):
    return OVR_BASE + OVR_PER_TIER * (lvl - 1)

# Precomputed ovr_strength by level (index 1..32); avoids ~1M function calls per solve.
_OVR_STR = [0.0] + [ovr_strength(l) for l in range(1, 33)]

MODULES = {
    'CMP': dict(name='Compiler',     cost=0,   prod={},           cons={}, special='compiler'),
    'PWR': dict(name='Power Supply', cost=10,  prod={'power':5},  cons={}),
    'CRW': dict(name='Crawler',      cost=15,  prod={'data':3},   cons={}),
    'RAM': dict(name='RAM Module',   cost=40,  prod={'memory':3}, cons={'power':2,'data':1}),
    'DB':  dict(name='Database',     cost=200, prod={'bits':3},   cons={'memory':3,'data':2,'power':1}),
    'OVR': dict(name='Overclocker',  cost=150, prod={},           cons={'power':4}, special='overclock'),
    'SAT': dict(name='Satellite Uplink', cost=150, prod={'data':5,'bits':4}, cons={'power':4},
                corner_only=True),
}

RESOURCES = ['power', 'data', 'memory', 'bits']
BALANCE   = ['power', 'data', 'memory']

ALLOWED = ['PWR', 'CRW', 'RAM', 'DB', 'OVR', 'SAT']

_SPECIAL = {t: m.get('special') for t, m in MODULES.items()}
_CORNER_ONLY = {t: bool(m.get('corner_only')) for t, m in MODULES.items()}
_RES_IDX = {'power': 0, 'data': 1, 'memory': 2, 'bits': 3}
_PROD_L = {t: [(_RES_IDX[r], a) for r, a in m['prod'].items()] for t, m in MODULES.items()}
_CONS_L = {t: [(_RES_IDX[r], a) for r, a in m['cons'].items()] for t, m in MODULES.items()}

EPS = 1e-6
_CACHE_CAP = 200_000
_EVAL_CAP = 5_000_000          # in-memory guard on the this-run new-entry dict
_PREFETCH_CAP = 500_000        # rows pulled into RAM per config at run start
_LEDGER = {}                   # prefetched entries for the current config (canon_key -> value)
_LEDGER_NEW = {}               # entries discovered this run (canon_key -> value)
_LEDGER_LOADED = False
_LEDGER_SIG = None             # config the in-RAM dicts belong to
_TCODE = {t: i + 1 for i, t in enumerate(MODULES)}


# --- symmetry-canonical grid keys --------------------------------------------
# evaluate() is invariant under the grid's symmetry group (flips / rotations),
# so we store each layout under the lexicographically smallest of its symmetric
# encodings. One row then covers 4 (rectangular) to 8 (square) layouts, and the
# search hits the cache even on a rotation it never literally generated.
_SYM_CACHE = {}


def _symmetries(w, h):
    got = _SYM_CACHE.get((w, h))
    if got is not None:
        return got
    n = w * h
    maps = [
        lambda r, c: (r, c),            # identity
        lambda r, c: (r, w - 1 - c),    # horizontal flip
        lambda r, c: (h - 1 - r, c),    # vertical flip
        lambda r, c: (h - 1 - r, w - 1 - c),  # 180 deg
    ]
    if w == h:
        maps += [
            lambda r, c: (c, r),                    # transpose
            lambda r, c: (c, h - 1 - r),            # 90 deg
            lambda r, c: (h - 1 - c, r),            # 270 deg
            lambda r, c: (h - 1 - c, w - 1 - r),    # anti-transpose
        ]
    perms = []
    for f in maps:
        perm = [0] * n
        for r in range(h):
            for c in range(w):
                orr, occ = f(r, c)
                perm[r * w + c] = orr * w + occ
        perms.append(perm)
    _SYM_CACHE[(w, h)] = perms
    return perms


def _pack_grid(grid):
    # 1 byte per cell: 0 = empty, else (type_code << 5) | level  (level <= 20 < 32)
    b = bytearray(len(grid))
    for i, c in enumerate(grid):
        if c is not None:
            b[i] = (_TCODE[c[0]] << 5) | (c[1] & 0x1F)
    return bytes(b)


def _canon_key(ctx, grid):
    # A cell's packed byte depends only on its contents, not its position, so a
    # symmetry just permutes the base packing. Pack once, then permute bytes --
    # far cheaper than re-packing the grid 8 times (this is a hot-loop call).
    base = _pack_grid(grid)
    best = base  # _symmetries()[0] is the identity
    for perm in _symmetries(ctx.w, ctx.h)[1:]:
        k = bytes([base[j] for j in perm])
        if k < best:
            best = k
    return best


def _key_hash(ctx, grid):
    # 8-byte stable digest of the canonical key. The ledger only guides search
    # (every reported/champion grid is re-scored by evaluate()), so the
    # astronomically-rare 64-bit collision can at worst nudge a search step --
    # never corrupt an output -- and it cuts the stored key from ~36 bytes to 8.
    return hashlib.blake2b(_canon_key(ctx, grid), digest_size=8).digest()


def _sig_str(ctx):
    return '%dx%d|%s|%d|%d' % (ctx.w, ctx.h, repr(round(ctx.mult, 9)),
                               ctx.max_level, int(ctx.uncap))

INFEASIBLE_PEN = 1e12
DEFICIT_W = 1e3
BELOW_TARGET_PEN = 1e8
SHORTFALL_W = 1e4


def _work_cores():
    return max(1, (os.cpu_count() or 2) - 2)


def corner_set(w, h):
    return {0, w - 1, (h - 1) * w, w * h - 1}


def _neighbors_list(w, h):
    nb = []
    for i in range(w * h):
        r, c = divmod(i, w)
        out = []
        if r > 0:          out.append(i - w)
        if r < h - 1:      out.append(i + w)
        if c > 0:          out.append(i - 1)
        if c < w - 1:      out.append(i + 1)
        nb.append(out)
    return nb


class Ctx:
    __slots__ = ('w', 'h', 'n', 'nb', 'corners', 'objective', 'mult',
                 'max_level', 'uncap', 'cache')

    def __init__(self, cfg):
        self.w = cfg.w
        self.h = cfg.h
        self.n = cfg.w * cfg.h
        self.nb = _neighbors_list(cfg.w, cfg.h)
        self.corners = corner_set(cfg.w, cfg.h)
        self.objective = cfg.objective
        self.mult = cfg.bits_mult
        self.max_level = cfg.max_level
        self.uncap = cfg.uncap_pwr
        self.cache = {}

    def type_max(self, t):
        return PWR_ABS_MAX if (t == 'PWR' and self.uncap) else self.max_level


def _type_merges(t, levels):
    return sum(int(LVL_WEIGHT ** (lvl - 1)) - 1 for lvl in levels)


def _neighbor_bonuses(ctx, grid):
    nb = ctx.nb
    ovr_adj = [0.0] * ctx.n; same_adj = [0] * ctx.n
    for i, cell in enumerate(grid):
        if not cell:
            continue
        t, lvl = cell
        if _SPECIAL[t] == 'overclock':
            s = _OVR_STR[lvl]
            for n in nb[i]:
                ovr_adj[n] += s
        for n in nb[i]:
            o = grid[n]
            if o and o[0] == t:
                same_adj[i] += 1
    return ovr_adj, same_adj


def evaluate(ctx, grid):
    ovr_adj, same_adj = _neighbor_bonuses(ctx, grid)
    corners = ctx.corners

    totals = [0.0, 0.0, 0.0, 0.0]
    base_equiv = {}
    tiers_by_type = {}
    misplaced = 0

    for i, cell in enumerate(grid):
        if not cell:
            continue
        t, lvl = cell
        if _SPECIAL[t] == 'compiler':
            continue
        tiers_by_type.setdefault(t, []).append(lvl)
        if _CORNER_ONLY[t] and i not in corners:
            misplaced += 1
            base_equiv[t] = base_equiv.get(t, 0.0) + LVL_WEIGHT ** (lvl - 1)
            continue
        out_mult  = 1 + (LVL_OUT - 1) * (lvl - 1)
        cons_mult = out_mult if SCALE_CONS_WITH_LEVEL else 1.0
        boost = 1.0 + ovr_adj[i] + 0.10 * same_adj[i]

        for ri, amt in _PROD_L[t]:
            totals[ri] += amt * out_mult * boost
        for ri, amt in _CONS_L[t]:
            totals[ri] -= amt * cons_mult

        base_equiv[t] = base_equiv.get(t, 0.0) + LVL_WEIGHT ** (lvl - 1)

    totals[3] *= ctx.mult

    cost = 0.0
    cost_by_type = {}
    merges = 0
    for t, levels in tiers_by_type.items():
        ct = MODULES[t]['cost'] * sum(LVL_COST_SCALE ** (lvl - 1) for lvl in levels)
        cost_by_type[t] = ct
        cost += ct
        merges += _type_merges(t, levels)

    feasible = (totals[0] >= -EPS and totals[1] >= -EPS and totals[2] >= -EPS
                and misplaced == 0)
    totals_d = {'power': totals[0], 'data': totals[1], 'memory': totals[2], 'bits': totals[3]}
    return dict(totals=totals_d, base_equiv=base_equiv, cost=cost, cost_by_type=cost_by_type,
                merges=merges, feasible=feasible, bits=totals[3], misplaced=misplaced,
                cells_used=sum(1 for c in grid if c))


def cell_bonuses(ctx, grid):
    ovr, same = _neighbor_bonuses(ctx, grid)
    out = []
    for i, cell in enumerate(grid):
        if not cell:
            out.append(0); continue
        out.append(round(ovr[i] * 100 + same[i] * 10))
    return out


def _fast_eval(ctx, grid):
    if _LEDGER_SIG != _sig_str(ctx):
        _prefetch_ledger(ctx)
    lk = _key_hash(ctx, grid)
    hit = _LEDGER.get(lk)
    if hit is None:
        hit = _LEDGER_NEW.get(lk)
    if hit is not None:
        return hit
    res = _fast_eval_raw(ctx, grid)
    if len(_LEDGER_NEW) < _EVAL_CAP:
        _LEDGER_NEW[lk] = res
    return res


def _fast_eval_raw(ctx, grid):
    ovr_adj, same_adj = _neighbor_bonuses(ctx, grid)
    corners = ctx.corners
    totals = [0.0, 0.0, 0.0, 0.0]
    misplaced = 0
    tiers = {}
    for i, cell in enumerate(grid):
        if not cell:
            continue
        t, lvl = cell
        if _SPECIAL[t] == 'compiler':
            continue
        tiers.setdefault(t, []).append(lvl)
        if _CORNER_ONLY[t] and i not in corners:
            misplaced += 1
            continue
        out_mult = 1 + (LVL_OUT - 1) * (lvl - 1)
        cons_mult = out_mult if SCALE_CONS_WITH_LEVEL else 1.0
        boost = 1.0 + ovr_adj[i] + 0.10 * same_adj[i]
        for ri, amt in _PROD_L[t]:
            totals[ri] += amt * out_mult * boost
        for ri, amt in _CONS_L[t]:
            totals[ri] -= amt * cons_mult
    bits = totals[3] * ctx.mult
    merges = sum(_type_merges(t, lv) for t, lv in tiers.items())
    feasible = misplaced == 0 and totals[0] >= -EPS and totals[1] >= -EPS and totals[2] >= -EPS
    deficit = 0.0
    if totals[0] < 0: deficit -= totals[0]
    if totals[1] < 0: deficit -= totals[1]
    if totals[2] < 0: deficit -= totals[2]
    return feasible, bits, merges, deficit


def _score_raw(ctx, grid, target):
    if ctx.objective == 'merges':
        feasible, bits, merges, deficit = _fast_eval(ctx, grid)
        if not feasible:
            return INFEASIBLE_PEN + deficit * DEFICIT_W + merges, None
        if bits < target - EPS:
            return BELOW_TARGET_PEN + (target - bits) * SHORTFALL_W + merges, None
        return merges, None
    ev = evaluate(ctx, grid)
    obj = ev['cost']
    if not ev['feasible']:
        deficit = sum(-ev['totals'][r] for r in BALANCE if ev['totals'][r] < 0)
        return INFEASIBLE_PEN + deficit * DEFICIT_W + obj, ev
    if ev['bits'] < target - EPS:
        return BELOW_TARGET_PEN + (target - ev['bits']) * SHORTFALL_W + obj, ev
    return obj, ev


def score(ctx, grid, target):
    cache = ctx.cache
    key = (tuple(grid), target)
    hit = cache.get(key)
    if hit is not None:
        return hit
    res = _score_raw(ctx, grid, target)
    if len(cache) < _CACHE_CAP:
        cache[key] = res
    return res


def random_cell(ctx, rng, idx):
    if rng.random() < 0.20:
        return None
    corners = ctx.corners
    choices = [t for t in ALLOWED if not _CORNER_ONLY[t] or idx in corners]
    t = rng.choice(choices)
    tm = ctx.type_max(t)
    r = rng.random()
    lvl = 1 + int(r * r * tm)
    return (t, max(1, min(lvl, tm)))


def _undo_corner_violation(ctx, g, i, j):
    corners = ctx.corners
    for k in (i, j):
        if g[k] and _CORNER_ONLY[g[k][0]] and k not in corners:
            g[i], g[j] = g[j], g[i]
            return


def _mutate_into(ctx, g, rng):
    n = ctx.n
    for _ in range(rng.choice([1, 1, 1, 2])):
        i = rng.randrange(n)
        if g[i] and g[i][0] == 'CMP':
            j = rng.randrange(n)
            g[i], g[j] = g[j], g[i]
            _undo_corner_violation(ctx, g, i, j)
            continue
        roll = rng.random()
        if roll < 0.55:
            g[i] = random_cell(ctx, rng, i)
        elif roll < 0.80 and g[i]:
            t, lvl = g[i]
            lvl = max(1, min(ctx.type_max(t), lvl + rng.choice([-1, 1])))
            g[i] = (t, lvl)
        else:
            j = rng.randrange(n)
            g[i], g[j] = g[j], g[i]
            _undo_corner_violation(ctx, g, i, j)


def mutate(ctx, grid, rng):
    g = list(grid)
    _mutate_into(ctx, g, rng)
    return g


def anneal(ctx, target, seed, rng, iters=6000, t0=8000.0):
    cur = list(seed)
    cur_s, _ = score(ctx, cur, target)
    best, best_s = list(cur), cur_s
    ratio = (1e-2 / t0) ** (1.0 / max(1, iters))
    T = t0
    for _ in range(iters):
        T *= ratio
        cand = mutate(ctx, cur, rng)
        cand_s, _ = score(ctx, cand, target)
        if cand_s < cur_s or rng.random() < math.exp(min(0.0, (cur_s - cand_s) / T)):
            cur, cur_s = cand, cand_s
            if cur_s < best_s:
                best, best_s = list(cur), cur_s
    return best, best_s


def _cell_options(ctx, i):
    corners = ctx.corners
    opts = [None]
    for t in ALLOWED:
        if _CORNER_ONLY[t] and i not in corners:
            continue
        for lvl in range(1, ctx.type_max(t) + 1):
            opts.append((t, lvl))
    return opts


def _cell_contrib(ctx, g, n):
    # Returns [power, data, memory, bits] contribution of cell n (index-based to
    # stay off the dict-building hot path; this runs ~1M times per solve).
    d = [0.0, 0.0, 0.0, 0.0]
    cell = g[n]
    if not cell:
        return d
    t, lvl = cell
    if _SPECIAL[t] == 'compiler':
        return d
    if _CORNER_ONLY[t] and n not in ctx.corners:
        return d
    ovr = 0.0; same = 0
    for k in ctx.nb[n]:
        o = g[k]
        if not o:
            continue
        if _SPECIAL[o[0]] == 'overclock':
            ovr += _OVR_STR[o[1]]
        if o[0] == t:
            same += 1
    out_mult  = 1 + (LVL_OUT - 1) * (lvl - 1)
    cons_mult = out_mult if SCALE_CONS_WITH_LEVEL else 1.0
    boost = 1.0 + ovr + 0.10 * same
    for ri, amt in _PROD_L[t]:
        d[ri] += amt * out_mult * boost
    for ri, amt in _CONS_L[t]:
        d[ri] -= amt * cons_mult
    return d


def _cell_cost(cell):
    if not cell:
        return 0.0
    t, lvl = cell
    if _SPECIAL[t] == 'compiler':
        return 0.0
    return MODULES[t]['cost'] * (LVL_COST_SCALE ** (lvl - 1))


def _cell_merges(cell):
    if not cell:
        return 0
    t, lvl = cell
    if _SPECIAL[t] == 'compiler':
        return 0
    return int(LVL_WEIGHT ** (lvl - 1)) - 1


def _cell_misplaced(ctx, cell, i):
    return 1 if (cell and _CORNER_ONLY[cell[0]] and i not in ctx.corners) else 0


def _grid_state(ctx, g):
    # Full-grid tally as [power, data, memory, bits_raw] (bits NOT multiplied by
    # ctx.mult), so _scan_cell can carry the running state and skip a fresh
    # O(n) evaluate() per cell.
    ev = evaluate(ctx, g); t = ev['totals']
    totals = [t['power'], t['data'], t['memory'], t['bits'] / ctx.mult]
    return {'totals': totals, 'cost': ev['cost'],
            'merges': ev['merges'], 'misplaced': ev['misplaced']}


def _scan_cell(ctx, g, i, target, state=None):
    if state is None:
        state = _grid_state(ctx, g)
    S = [i] + ctx.nb[i]
    bg = list(state['totals'])
    for n in S:
        c = _cell_contrib(ctx, g, n)
        bg[0] -= c[0]; bg[1] -= c[1]; bg[2] -= c[2]; bg[3] -= c[3]
    old = g[i]
    bg_cost = state['cost'] - _cell_cost(old)
    bg_merges = state['merges'] - _cell_merges(old)
    bg_misplaced = state['misplaced'] - _cell_misplaced(ctx, old, i)

    best_opt, best_s, best_state = old, None, None
    for opt in [old] + [o for o in _cell_options(ctx, i) if o != old]:
        g[i] = opt
        tot = list(bg)
        for n in S:
            c = _cell_contrib(ctx, g, n)
            tot[0] += c[0]; tot[1] += c[1]; tot[2] += c[2]; tot[3] += c[3]
        cost = bg_cost + _cell_cost(opt)
        merges = bg_merges + _cell_merges(opt)
        misplaced = bg_misplaced + _cell_misplaced(ctx, opt, i)
        bits = tot[3] * ctx.mult
        obj = merges if ctx.objective == 'merges' else cost
        feasible = misplaced == 0 and tot[0] >= -EPS and tot[1] >= -EPS and tot[2] >= -EPS
        if not feasible:
            deficit = 0.0
            if tot[0] < 0: deficit -= tot[0]
            if tot[1] < 0: deficit -= tot[1]
            if tot[2] < 0: deficit -= tot[2]
            s = INFEASIBLE_PEN + deficit * DEFICIT_W + obj
        elif bits < target - EPS:
            s = BELOW_TARGET_PEN + (target - bits) * SHORTFALL_W + obj
        else:
            s = obj
        if best_s is None or s < best_s - 1e-9:
            best_s, best_opt = s, opt
            best_state = {'totals': list(tot), 'cost': cost,
                          'merges': merges, 'misplaced': misplaced}
    g[i] = old
    return best_opt, best_s, best_state


def polish(ctx, grid, target):
    n = ctx.n; corners = ctx.corners
    g = list(grid)
    improved = True
    while improved:
        improved = False
        state = _grid_state(ctx, g)
        for i in range(n):
            if g[i] and g[i][0] == 'CMP':
                continue
            best_opt, _, best_state = _scan_cell(ctx, g, i, target, state)
            if best_opt != g[i]:
                g[i] = best_opt
                state = best_state
                improved = True
        ci = next((k for k in range(n) if g[k] and g[k][0] == 'CMP'), None)
        if ci is not None:
            base_s, _ = score(ctx, g, target)
            best_j, best_s = ci, base_s
            for j in range(n):
                if j == ci:
                    continue
                trial = list(g); trial[ci], trial[j] = trial[j], trial[ci]
                if any(trial[k] and _CORNER_ONLY[trial[k][0]]
                       and k not in corners for k in (ci, j)):
                    continue
                s, _ = score(ctx, trial, target)
                if s < best_s - 1e-9:
                    best_s, best_j = s, j
            if best_j != ci:
                g[ci], g[best_j] = g[best_j], g[ci]
                improved = True
    return g


def _perturb(ctx, grid, rng, kicks):
    g = list(grid)
    for _ in range(kicks):
        _mutate_into(ctx, g, rng)
    return g


def _crossover(ctx, a, b, rng):
    w, h, n = ctx.w, ctx.h, ctx.n
    r0, r1 = (sorted(rng.sample(range(h), 2)) if h > 1 else (0, 0))
    c0, c1 = (sorted(rng.sample(range(w), 2)) if w > 1 else (0, 0))
    child = list(b)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            i = r * w + c
            child[i] = a[i]
    cmps = [i for i, x in enumerate(child) if x and x[0] == 'CMP']
    if not cmps:
        child[rng.randrange(n)] = ('CMP', 1)
    else:
        for i in cmps[1:]:
            child[i] = None
    for i, x in enumerate(child):
        if x and _CORNER_ONLY[x[0]] and i not in ctx.corners:
            child[i] = None
    return child


def memetic(ctx, target, base_seed=1234, pop_size=24, gens=400, iters=3000, incumbent=None,
            incumbents=None, deadline=None, stop_ev=None):
    import time as _time
    def _stop():
        return ((stop_ev is not None and stop_ev.is_set())
                or (deadline is not None and _time.time() >= deadline))
    rng = random.Random(base_seed)
    n = ctx.n
    corners = sorted(ctx.corners)
    pop = []
    seeds = list(incumbents) if incumbents else []
    if incumbent is not None:
        seeds.append(incumbent)
    for inc in seeds:
        if not inc or len(inc) != n:
            continue
        g = polish(ctx, list(inc), target)
        pop.append((score(ctx, g, target)[0], g))
    for r in range(pop_size):
        if _stop() and pop:
            break
        s = [random_cell(ctx, rng, k) for k in range(n)]
        s[corners[r] if r < len(corners) else rng.randrange(n)] = ('CMP', 1)
        g = polish(ctx, anneal(ctx, target, s, rng, iters=iters)[0], target)
        pop.append((score(ctx, g, target)[0], g))
    pop.sort(key=lambda x: x[0])
    if len(pop) > pop_size:
        pop = pop[:pop_size]
    for _ in range(gens):
        if _stop():
            break
        pa = min(rng.sample(pop, 3), key=lambda x: x[0])[1]
        pb = min(rng.sample(pop, 3), key=lambda x: x[0])[1]
        child = polish(ctx, _perturb(ctx, _crossover(ctx, pa, pb, rng), rng, rng.choice([0, 1, 2])), target)
        cs = score(ctx, child, target)[0]
        if cs < pop[-1][0]:
            pop[-1] = (cs, child)
            pop.sort(key=lambda x: x[0])
    return pop[0][1]


def optimize(ctx, target, restarts=150, iters=6000, base_seed=1234, incumbent=None,
             ils_iters=400, t0=8000.0, incumbents=None, deadline=None, stop_event=None):
    import time as _time
    def _stop():
        return ((stop_event is not None and stop_event.is_set())
                or (deadline is not None and _time.time() >= deadline))
    n = ctx.n
    best, best_s = None, float('inf')
    seeds = list(incumbents) if incumbents else []
    if incumbent is not None:
        seeds.append(incumbent)
    for inc in seeds:
        if not inc or len(inc) != n:
            continue
        g = polish(ctx, list(inc), target)
        s = score(ctx, g, target)[0]
        if s < best_s:
            best, best_s = g, s
    corners = sorted(ctx.corners)
    for r in range(restarts):
        if _stop():
            break
        rng = random.Random(base_seed + r * 7919)
        seed = [random_cell(ctx, rng, i) for i in range(n)]
        cpos = corners[r] if r < len(corners) else rng.randrange(n)
        seed[cpos] = ('CMP', 1)
        g = polish(ctx, anneal(ctx, target, seed, rng, iters=iters, t0=t0)[0], target)
        s, _ = score(ctx, g, target)
        if s < best_s:
            best, best_s = g, s
    if ils_iters and best is not None:
        rng = random.Random(base_seed + 990331)
        cur, cur_s = best, best_s
        for _ in range(ils_iters):
            if _stop():
                break
            cand = polish(ctx, _perturb(ctx, cur, rng, rng.choice([2, 3, 4, 5])), target)
            cs, _ = score(ctx, cand, target)
            if cs < best_s:
                best, best_s, cur, cur_s = cand, cs, cand, cs
            elif cs <= cur_s * 1.02:
                cur, cur_s = cand, cs
    return best


SVG_ICON = {
    'CMP': '<polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/>',
    'PWR': '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    'CRW': '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    'RAM': '<path d="M6 19v-3"/><path d="M10 19v-3"/><path d="M14 19v-3"/><path d="M18 19v-3"/><path d="M8 11V9"/><path d="M16 11V9"/><path d="M12 11V9"/><path d="M2 15h20"/><path d="M2 7a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v1.1a2 2 0 0 0 0 3.837V17a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-5.1a2 2 0 0 0 0-3.837Z"/>',
    'DB':  '<line x1="22" x2="2" y1="12" y2="12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><line x1="6" x2="6.01" y1="16" y2="16"/><line x1="10" x2="10.01" y1="16" y2="16"/>',
    'OVR': '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    'SAT': '<path d="M4 10a7.31 7.31 0 0 0 10 10Z"/><path d="m9 15 3-3"/><path d="M17 13a6 6 0 0 0-6-6"/><path d="M21 13A10 10 0 0 0 11 3"/>',
}

def _svg(t, cls='icon'):
    return ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            'class="%s">%s</svg>' % (cls, SVG_ICON.get(t, '')))

TIER_COLORS = {
    1: '#565c66', 2: '#3b82f6', 3: '#0ea5b7', 4: '#17a398', 5: '#6cc93f',
    6: '#eab308', 7: '#f97316', 8: '#ef4444', 9: '#ec4899', 10: '#a855f7',
    11: '#8b5cf6', 12: '#d946ef',
}

def _tier_color(lvl):
    if lvl in TIER_COLORS:
        return TIER_COLORS[lvl]
    return TIER_COLORS[max(TIER_COLORS)] if lvl > max(TIER_COLORS) else TIER_COLORS[1]

def _darken(hexcol, f):
    h = hexcol.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return '#%02x%02x%02x' % (int(r * f), int(g * f), int(b * f))

def _tier_bg(lvl):
    c = _tier_color(lvl)
    return 'linear-gradient(180deg,%s,%s)' % (c, _darken(c, 0.66))

def _tile_html(cell, ob):
    if not cell:
        return '<div class="tile empty"></div>'
    t, lvl = cell
    if t == 'CMP':
        return ('<div class="tile compiler" title="Compiler (source)">'
                '%s<div class="lbl">Compiler</div></div>' % _svg('CMP'))
    bonus_html = '<div class="bonus"><span class="b-out">+%d%%</span></div>' % ob if ob else ''
    return ('<div class="tile" style="background:%s" title="%s (lvl %d)">'
            '<div class="badge">%d</div>%s<div class="lbl">%s</div>%s</div>'
            % (_tier_bg(lvl), MODULES[t]['name'], lvl, lvl, _svg(t),
               MODULES[t]['name'], bonus_html))

_PAGE_CSS = """
  *{box-sizing:border-box}
  body{background:#0b0c0f;color:#e6e8ec;margin:0;padding:36px;
       font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
       display:flex;flex-direction:column;align-items:center}
  h1{font-size:15px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
     color:#9aa0aa;margin:0 0 20px}
  .main{display:flex;gap:30px;align-items:flex-start;margin-bottom:30px;flex-wrap:wrap}
  .grid{display:grid;gap:11px;width:max-content}
  .summary{min-width:210px;background:#14161b;border:1px solid #23262d;
           border-radius:12px;padding:14px 18px}
  .summary h2{font-size:11px;font-weight:700;text-transform:uppercase;
              letter-spacing:.06em;color:#8b909a;margin:0 0 12px}
  .sgroup{margin-bottom:13px}
  .sgroup:last-child{margin-bottom:0}
  .sname{font-size:13px;font-weight:700;color:#eef1f4;margin-bottom:5px;
         display:flex;justify-content:space-between;gap:14px}
  .stot{color:#8b909a;font-weight:600}
  .srow{display:flex;justify-content:space-between;gap:14px;
        font-size:12px;color:#c3c8d0;padding:2px 0}
  .srow .tl{display:flex;align-items:center;gap:7px}
  .srow .sw{width:10px;height:10px;border-radius:3px;flex:0 0 auto}
  .srow .ct{color:#9aa0aa}
  .tile{position:relative;width:96px;height:96px;border-radius:16px;
        background:linear-gradient(180deg,#41474f,#2f343b);
        display:flex;flex-direction:column;align-items:center;justify-content:center;
        gap:6px;color:#fff;box-shadow:inset 0 1px 0 rgba(255,255,255,.12)}
  .tile.empty{background:transparent;border:2px dashed #3a3f47;box-shadow:none}
  .tile.compiler{border:2px solid #caa14a;
        background:linear-gradient(180deg,#3a352a,#26221a)}
  .icon{width:34px;height:34px;filter:drop-shadow(0 1px 2px rgba(0,0,0,.4))}
  .lbl{font-size:10px;font-weight:600;color:#fff;letter-spacing:.02em;
       text-align:center;line-height:1.1;max-width:88px;
       text-shadow:0 1px 2px rgba(0,0,0,.55)}
  .tile.compiler .lbl{color:#e3c98a;text-shadow:none}
  .badge{position:absolute;top:-7px;right:-7px;min-width:23px;height:23px;padding:0 5px;
         border-radius:12px;background:#15171c;border:1px solid #2b2f36;
         color:#fff;font-size:12px;font-weight:700;
         display:flex;align-items:center;justify-content:center}
  .bonus{position:absolute;left:9px;bottom:6px;font-size:10px;font-weight:700;
         display:flex;gap:5px;text-shadow:0 1px 2px rgba(0,0,0,.75)}
  .b-out{color:#eafff0}
  .stats{display:flex;flex-wrap:wrap;gap:12px;max-width:640px}
  .card{background:#14161b;border:1px solid #23262d;border-radius:12px;
        padding:12px 16px;min-width:120px}
  .card .k{font-size:11px;color:#8b909a;text-transform:uppercase;letter-spacing:.05em}
  .card .v{font-size:20px;font-weight:700;margin-top:3px}
  .bal{margin-top:20px;max-width:640px;font-size:13px}
  .bal .row{display:flex;justify-content:space-between;padding:5px 0;
            border-bottom:1px solid #1b1e24}
  .pos{color:#4ade80}.neg{color:#f87171}.goal{color:#60a5fa}
"""

def _summary_html(grid):
    present = {}
    for c in grid:
        if not c:
            continue
        t, lvl = c
        if _SPECIAL[t] == 'compiler':
            continue
        present.setdefault(t, Counter())[lvl] += 1
    order = [t for t in ALLOWED if t in present] + \
            [t for t in present if t not in ALLOWED]
    groups = ''
    for t in order:
        cnt = present[t]
        rows = ''.join(
            '<div class="srow"><span class="tl">'
            '<span class="sw" style="background:%s"></span>Tier %d</span>'
            '<span class="ct">&times;%d</span></div>'
            % (_tier_color(lvl), lvl, cnt[lvl]) for lvl in sorted(cnt, reverse=True))
        groups += ('<div class="sgroup"><div class="sname">%s'
                   '<span class="stot">&times;%d</span></div>%s</div>'
                   % (MODULES[t]['name'], sum(cnt.values()), rows))
    return '<div class="summary"><h2>Modules</h2>%s</div>' % groups


def build_page(ctx, grid, target):
    ev = evaluate(ctx, grid)
    bon = cell_bonuses(ctx, grid)
    tiles = ''.join(_tile_html(grid[i], bon[i]) for i in range(ctx.n))
    ok = ev['feasible'] and ev['bits'] >= target - EPS
    cards = [
        ('bits/sec', '%.1f' % ev['bits'], 'goal'),
        ('target', '%g' % target, 'pos' if ok else 'neg'),
        ('cells used', '%d/%d' % (ev['cells_used'], ctx.n), ''),
        ('manual merges', '%d' % ev['merges'], 'goal' if ctx.objective == 'merges' else ''),
    ]
    cards_html = ''.join(
        '<div class="card"><div class="k">%s</div><div class="v %s">%s</div></div>'
        % (k, cls, v) for k, v, cls in cards)
    rows = ''
    for res in RESOURCES:
        v = ev['totals'][res]
        cls = 'goal' if res == 'bits' else ('neg' if v < -EPS else 'pos')
        rows += ('<div class="row"><span>%s</span><span class="%s">%+.2f/sec</span></div>'
                 % (res, cls, v))
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<title>Grid layout</title><style>%s</style></head><body>'
        '<h1>Best layout &mdash; %dx%d grid, target %g bits/sec</h1>'
        '<div class="main">'
        '<div class="grid" style="grid-template-columns:repeat(%d,96px)">%s</div>'
        '%s'
        '</div>'
        '<div class="stats">%s</div>'
        '<div class="bal">%s</div>'
        '</body></html>'
        % (_PAGE_CSS, ctx.w, ctx.h, target, ctx.w, tiles,
           _summary_html(grid), cards_html, rows))


def write_and_open_html(ctx, grid, target):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grid_layout.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(build_page(ctx, grid, target))
    try:
        if os.name == 'nt':
            os.startfile(path)
    except Exception:
        pass
    return path


def report(ctx, grid, target):
    ev = evaluate(ctx, grid)
    print("\n" + "=" * 60)
    print(f"  BEST LAYOUT  (target {target:g} bits/sec)")
    print("=" * 60)
    html_path = write_and_open_html(ctx, grid, target)
    print(f"\n  Grid rendered to:  {html_path}")
    print("\n  Resource balance (net /sec):")
    for res in RESOURCES:
        v = ev['totals'][res]
        tag = "  <-- GOAL" if res == 'bits' else ("  (deficit!)" if v < -EPS else "")
        print(f"     {res:<7}: {v:+8.2f}{tag}")
    ok = ev['feasible'] and ev['bits'] >= target - EPS
    print(f"\n  bits/sec produced : {ev['bits']:.2f}   "
          f"({'MEETS' if ok else 'BELOW'} target {target:g})")
    print(f"  cells used        : {ev['cells_used']}/{ctx.n}")
    star = "  <-- MINIMISED" if ctx.objective == 'merges' else ""
    print(f"  manual merges     : {ev['merges']}{star}")
    print(f"  total token cost  : {ev['cost']:.0f}")
    print("=" * 60 + "\n")


_SAVED = {'target': 1220.0, 'w': 6, 'h': 6, 'mult': 1.6, 'maxlvl': 10, 'time_limit': 180,
          'uncap_pwr': False, 'objective': 'merges'}
_CHAMPIONS = {}
_RUNS = {}


def _ortools_available():
    try:
        from ortools.sat.python import cp_model  # noqa: F401
        return True
    except Exception:
        return False


def _cpsat_optimize(ctx, target, time_limit, hint=None, stop_event=None):
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None
    SO = 1000; SS = 1000
    N = ctx.n; nb = ctx.nb; corners = ctx.corners
    PLACE = ['PWR', 'CRW', 'RAM', 'DB', 'OVR', 'SAT']
    Lmax = max(ctx.type_max(t) for t in PLACE)
    OUT = [round(SO * (1 + (LVL_OUT - 1) * (l - 1))) for l in range(1, Lmax + 1)]
    OVRSTR = [round(SS * ovr_strength(l)) for l in range(1, Lmax + 1)]
    RES = ['power', 'data', 'memory']
    mdl = cp_model.CpModel()

    def states(i):
        s = ['E', 'C']
        for t in PLACE:
            if _CORNER_ONLY[t] and i not in corners:
                continue
            for l in range(1, ctx.type_max(t) + 1):
                s.append((t, l))
        return s

    x = {}
    for i in range(N):
        vs = {s: mdl.NewBoolVar(f'x_{i}_{s}') for s in states(i)}
        mdl.AddExactlyOne(vs.values())
        x[i] = vs
    mdl.Add(sum(x[i]['C'] for i in range(N)) == 1)

    ovr_str = {p: sum(OVRSTR[s[1] - 1] * v for s, v in x[p].items()
                      if isinstance(s, tuple) and s[0] == 'OVR') for p in range(N)}
    same = {}
    for i in range(N):
        terms = []
        for p in nb[i]:
            for t in PLACE:
                it_i = [v for s, v in x[i].items() if isinstance(s, tuple) and s[0] == t]
                it_n = [v for s, v in x[p].items() if isinstance(s, tuple) and s[0] == t]
                if not it_i or not it_n:
                    continue
                a = mdl.NewBoolVar(f'and_{i}_{p}_{t}')
                bi, bn = sum(it_i), sum(it_n)
                mdl.Add(a <= bi); mdl.Add(a <= bn); mdl.Add(a >= bi + bn - 1)
                terms.append(a)
        same[i] = sum(terms) if terms else 0
    OVRMAX = max(OVRSTR)
    boost = {}
    for i in range(N):
        b = mdl.NewIntVar(SS, SS + 4 * OVRMAX + 4 * (SS // 10), f'boost_{i}')
        mdl.Add(b == SS + sum(ovr_str[p] for p in nb[i]) + (SS // 10) * same[i])
        boost[i] = b
    OUTMAX = max(OUT)
    prod_micro = {r: [] for r in RES + ['bits']}
    cons_scaled = {r: [] for r in RES}
    for i in range(N):
        for r in RES + ['bits']:
            praw = sum(MODULES[s[0]]['prod'].get(r, 0) * OUT[s[1] - 1] * v
                       for s, v in x[i].items() if isinstance(s, tuple))
            if isinstance(praw, int):
                continue
            pv = mdl.NewIntVar(0, 9 * OUTMAX, f'praw_{r}_{i}'); mdl.Add(pv == praw)
            pm = mdl.NewIntVar(0, 9 * OUTMAX * (SS + 4 * OVRMAX + 4 * (SS // 10)), f'pm_{r}_{i}')
            mdl.AddMultiplicationEquality(pm, [pv, boost[i]])
            prod_micro[r].append(pm)
        for r in RES:
            craw = sum(MODULES[s[0]]['cons'].get(r, 0) * OUT[s[1] - 1] * v
                       for s, v in x[i].items() if isinstance(s, tuple))
            if not isinstance(craw, int):
                cons_scaled[r].append(craw)
    for r in RES:
        mdl.Add(sum(prod_micro[r]) - SS * sum(cons_scaled[r]) >= 0)
    mult10 = round(ctx.mult * 10)
    mdl.Add(mult10 * sum(prod_micro['bits']) >= round(target * 10 * SO * SS))

    if ctx.objective == 'merges':
        obj = sum((int(LVL_WEIGHT ** (s[1] - 1)) - 1) * v
                  for i in range(N) for s, v in x[i].items() if isinstance(s, tuple))
    else:
        obj = sum(round(MODULES[s[0]]['cost'] * (LVL_COST_SCALE ** (s[1] - 1))) * v
                  for i in range(N) for s, v in x[i].items() if isinstance(s, tuple))
    mdl.Minimize(obj)

    can_clear = hasattr(mdl, 'ClearHints')
    _hinted = [False]

    def apply_hint(h):
        if h is None or (_hinted[0] and not can_clear):
            return
        if can_clear:
            mdl.ClearHints()
        for i in range(N):
            cell = h[i]
            st = 'E' if not cell else ('C' if _SPECIAL[cell[0]] == 'compiler' else cell)
            if st in x[i]:
                mdl.AddHint(x[i][st], 1)
        _hinted[0] = True

    def extract():
        grid = [None] * N
        for i in range(N):
            for s, v in x[i].items():
                if solver.Value(v):
                    grid[i] = ('CMP', 1) if s == 'C' else (None if s == 'E' else s)
        return grid

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = _work_cores()

    class _Stopper(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            if stop_event is not None and stop_event.is_set():
                self.StopSearch()

    # CP-SAT's solution callback only fires on *new* solutions, so a plain
    # Solve() ignores a stop request during the (often long) optimality-proving
    # tail. Slice the budget into short chunks, warm-starting each from the best
    # solution found, so Stop is honored within one slice.
    import time as _t
    end = _t.time() + float(time_limit)
    SLICE = 2.5
    best = hint
    found = False
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        remaining = end - _t.time()
        if remaining <= 1e-3:
            break
        apply_hint(best)
        solver.parameters.max_time_in_seconds = min(SLICE, remaining)
        status = solver.Solve(mdl, _Stopper())
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            best = extract(); found = True
            if status == cp_model.OPTIMAL:
                break                      # proven optimal -- nothing more to gain
        elif status == cp_model.INFEASIBLE:
            break                          # no solution exists; further slices won't help
    return best if found else None


def _worker(p):
    global _LEDGER_NEW
    _LEDGER_NEW = {}
    ctx = Ctx(p['cfg'])
    _prefetch_ledger(ctx)
    gens = max(80, p['restarts'] * 6)
    grid = memetic(ctx, p['target'], base_seed=p['seed'], pop_size=24,
                   gens=gens, iters=p['iters'], incumbents=p.get('incumbents'),
                   deadline=p.get('deadline'), stop_ev=p.get('stop_ev'))
    return (score(ctx, grid, p['target'])[0], grid, _LEDGER_NEW)


def optimize_parallel(ctx, cfg, target, restarts, iters, incumbents=None, seed_off=0,
                      deadline=None, stop_event=None):
    if getattr(sys, 'frozen', False):
        return optimize(ctx, target, restarts=min(restarts, 8), iters=iters,
                        base_seed=1234 + seed_off, incumbents=incumbents,
                        deadline=deadline, stop_event=stop_event)
    try:
        import multiprocessing as mp
        ncores = _work_cores()
    except Exception:
        ncores = 1
    ncores = max(1, min(ncores, max(1, restarts)))
    if ncores == 1:
        return optimize(ctx, target, restarts=restarts, iters=iters,
                        base_seed=1234 + seed_off, incumbents=incumbents,
                        deadline=deadline, stop_event=stop_event)
    base, extra = divmod(restarts, ncores)
    mpctx = mp.get_context('spawn')
    mgr = mp_stop = None
    try:
        if stop_event is not None:
            mgr = mpctx.Manager()
            mp_stop = mgr.Event()
    except Exception:
        mgr = mp_stop = None
    payloads = []
    for k in range(ncores):
        rs = base + (1 if k < extra else 0)
        if rs <= 0:
            continue
        payloads.append(dict(target=target, cfg=cfg,
                             restarts=rs, iters=iters, seed=1234 + seed_off + k * 100003,
                             incumbents=incumbents, deadline=deadline, stop_ev=mp_stop))
    bridge = {'run': True}
    watcher = None
    try:
        if stop_event is not None and mp_stop is not None:
            import threading, time as _t
            def _bridge():
                while bridge['run']:
                    if stop_event.is_set():
                        mp_stop.set(); return
                    _t.sleep(0.1)
            watcher = threading.Thread(target=_bridge, daemon=True)
            watcher.start()
        with mpctx.Pool(len(payloads)) as pool:
            results = [r for r in pool.map(_worker, payloads) if r]
        if not results:
            return optimize(ctx, target, restarts=restarts, iters=iters,
                            base_seed=1234 + seed_off, incumbents=incumbents,
                            deadline=deadline, stop_event=stop_event)
        for r in results:
            if len(r) > 2 and r[2]:
                _LEDGER_NEW.update(r[2])
        return min(results, key=lambda r: r[0])[1]
    except Exception:
        return optimize(ctx, target, restarts=restarts, iters=iters,
                        base_seed=1234 + seed_off, incumbents=incumbents,
                        deadline=deadline, stop_event=stop_event)
    finally:
        bridge['run'] = False
        if mgr is not None:
            try:
                mgr.shutdown()
            except Exception:
                pass


def solve_engine(ctx, cfg, target, time_limit, priors=None, stop_event=None, seed_off=0,
                 progress=None):
    import time as _time
    start = _time.time()
    deadline = start + time_limit
    priors = [g for g in (priors or []) if g is not None]
    obj = ctx.objective                      # 'merges' or 'cost' -- both keys in evaluate()
    def rank(g):
        ev = evaluate(ctx, g)
        feasible = ev['feasible'] and ev['bits'] >= target - EPS
        return (0 if feasible else 1, ev[obj])
    def stopped():
        return (stop_event is not None and stop_event.is_set()) or _time.time() >= deadline
    def report(g):
        if progress is not None and g is not None:
            try:
                progress(g)
            except Exception:
                pass
    par = optimize_parallel(ctx, cfg, target, 48, 3000, incumbents=priors, seed_off=seed_off,
                            deadline=deadline, stop_event=stop_event)
    pool = [g for g in ([par] + priors) if g is not None]
    best = min(pool, key=rank) if pool else None
    report(best)
    if not stopped():
        # give CP-SAT only the time that's actually left, so the total run
        # honors the requested limit instead of spending it twice.
        remaining = max(1.0, deadline - _time.time())
        cp = _cpsat_optimize(ctx, target, remaining, hint=best, stop_event=stop_event)
        if cp is not None and (best is None or rank(cp) < rank(best)):
            best = cp; report(best)
    it = 0
    while not stopped():
        cand = optimize(ctx, target, restarts=6, iters=3000, base_seed=1234 + seed_off + it * 7789,
                        incumbent=best, incumbents=priors, deadline=deadline, stop_event=stop_event)
        if best is None or rank(cand) < rank(best):
            best = cand; report(best)
        it += 1
    return best


_FROZEN = getattr(sys, 'frozen', False)


def _load_settings():
    return dict(_SAVED)


def _state_path():
    base = os.path.dirname(sys.executable) if _FROZEN else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'grid_optimizer_state.txt')


def _ledger_path():
    base = os.path.dirname(sys.executable) if _FROZEN else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'grid_optimizer_ledger.db')


def _ledger_sig():
    # Hash of every constant/module fact that _fast_eval depends on. If any of
    # these change, cached (feasible, bits, merges, deficit) tuples are stale.
    import hashlib
    parts = [LVL_OUT, LVL_WEIGHT, LVL_COST_SCALE, PWR_ABS_MAX,
             int(SCALE_CONS_WITH_LEVEL), OVR_BASE, OVR_PER_TIER]
    for t in sorted(MODULES):
        m = MODULES[t]
        parts.append((t, m['cost'],
                      tuple(sorted(m.get('prod', {}).items())),
                      tuple(sorted(m.get('cons', {}).items())),
                      m.get('special'), bool(m.get('corner_only'))))
    return hashlib.md5(repr(parts).encode('utf-8')).hexdigest()


_LEGACY_CLEANED = False
_BITS_SCALE = 1000        # bits stored as round(bits * this) so an int varint replaces an 8-byte REAL


# Compact archive of scanned grids. Only FEASIBLE grids are persisted -- an
# infeasible grid is cheap to recompute and rarely revisited across runs, so it
# stays in the per-run in-memory cache only. Layout choices that shrink the file:
#   * cfg table maps each config signature to a small int id (no repeated text)
#   * key is an 8-byte hash of the canonical grid, not the ~36-byte grid itself
#   * bits is a scaled int; feasible/deficit are implied (True / 0) so no columns
def _open_db():
    global _LEGACY_CLEANED
    try:
        import sqlite3
        con = sqlite3.connect(_ledger_path(), timeout=10)
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('PRAGMA synchronous=NORMAL')
        con.execute('CREATE TABLE IF NOT EXISTS cfg(id INTEGER PRIMARY KEY, sig TEXT UNIQUE)')
        con.execute('CREATE TABLE IF NOT EXISTS grids('
                    'c INTEGER, k BLOB, bits INTEGER, merges INTEGER, '
                    'PRIMARY KEY(c, k)) WITHOUT ROWID')
        con.execute('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)')
        want = _ledger_sig()
        row = con.execute("SELECT v FROM meta WHERE k='scoresig'").fetchone()
        constants_ok = row is None or row[0] == want
        if row is None:
            con.execute("INSERT INTO meta(k, v) VALUES('scoresig', ?)", (want,))
        elif not constants_ok:                     # scoring constants changed -> stale
            con.execute('DELETE FROM grids'); con.execute('DELETE FROM cfg')
            con.execute("UPDATE meta SET v=? WHERE k='scoresig'", (want,))
        con.commit()
        _migrate_legacy(con, constants_ok)
        if not _LEGACY_CLEANED:
            _LEGACY_CLEANED = True
            try:                                    # reclaim the old pickle cache, if present
                os.remove(os.path.splitext(_ledger_path())[0] + '.dat')
            except Exception:
                pass
        return con
    except Exception:
        return None


def _migrate_legacy(con, constants_ok):
    # One-time: fold an old-schema `ledger` table into the compact `grids` table,
    # keeping only feasible rows and re-hashing their keys, then reclaim the space.
    if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ledger'").fetchone():
        return
    try:
        if constants_ok:
            for (sig,) in con.execute('SELECT DISTINCT sig FROM ledger').fetchall():
                con.execute('INSERT OR IGNORE INTO cfg(sig) VALUES(?)', (sig,))
                cid = con.execute('SELECT id FROM cfg WHERE sig=?', (sig,)).fetchone()[0]
                last = b''
                while True:                          # keyset pagination keeps RAM bounded
                    page = con.execute(
                        'SELECT gkey, bits, merges FROM ledger '
                        'WHERE feasible=1 AND sig=? AND gkey>? ORDER BY gkey LIMIT 50000',
                        (sig, last)).fetchall()
                    if not page:
                        break
                    con.executemany(
                        'INSERT OR IGNORE INTO grids(c, k, bits, merges) VALUES(?, ?, ?, ?)',
                        [(cid, hashlib.blake2b(g, digest_size=8).digest(),
                          int(round(b * _BITS_SCALE)), int(m)) for g, b, m in page])
                    last = page[-1][0]
                    con.commit()
        con.execute('DROP TABLE ledger')
        con.commit()
        con.execute('VACUUM')
        con.commit()
    except Exception:
        pass


def _cfg_id(con, sig, create=False):
    row = con.execute('SELECT id FROM cfg WHERE sig=?', (sig,)).fetchone()
    if row:
        return row[0]
    if not create:
        return None
    con.execute('INSERT OR IGNORE INTO cfg(sig) VALUES(?)', (sig,))
    return con.execute('SELECT id FROM cfg WHERE sig=?', (sig,)).fetchone()[0]


def _prefetch_ledger(ctx):
    # Pull only the current config's rows into RAM; the DB keeps every other
    # config's rows on disk, untouched.
    global _LEDGER, _LEDGER_LOADED, _LEDGER_SIG
    _LEDGER_LOADED = True
    _LEDGER_SIG = _sig_str(ctx)
    _LEDGER = {}
    con = _open_db()
    if con is None:
        return
    try:
        cid = _cfg_id(con, _LEDGER_SIG)
        if cid is not None:
            cur = con.execute('SELECT k, bits, merges FROM grids WHERE c=? LIMIT ?',
                              (cid, _PREFETCH_CAP))
            _LEDGER = {k: (True, b / _BITS_SCALE, m, 0.0) for (k, b, m) in cur}
    except Exception:
        _LEDGER = {}
    finally:
        con.close()


def _save_ledger():
    if not _LEDGER_NEW:
        return
    con = _open_db()
    if con is None:
        return
    try:
        cid = _cfg_id(con, _LEDGER_SIG or '', create=True)
        rows = [(cid, k, int(round(v[1] * _BITS_SCALE)), int(v[2]))
                for k, v in _LEDGER_NEW.items() if v[0]]   # v[0] = feasible: skip the rest
        if rows:
            con.executemany('INSERT OR IGNORE INTO grids(c, k, bits, merges) '
                            'VALUES(?, ?, ?, ?)', rows)
            con.commit()
    except Exception:
        pass
    finally:
        con.close()


def _load_state():
    global _SAVED, _CHAMPIONS, _RUNS
    try:
        import ast
        with open(_state_path(), 'r', encoding='utf-8') as f:
            st = ast.literal_eval(f.read())
        _SAVED = st.get('saved', _SAVED)
        _CHAMPIONS = st.get('champions', _CHAMPIONS)
        _RUNS = st.get('runs', _RUNS)
    except Exception:
        pass


def _persist():
    try:
        with open(_state_path(), 'w', encoding='utf-8') as f:
            f.write(repr({'saved': _SAVED, 'champions': _CHAMPIONS, 'runs': _RUNS}))
    except Exception:
        pass


def _save_settings(d):
    global _SAVED
    _SAVED = dict(d)
    _persist()


def _next_seed_off(key):
    global _RUNS
    n = _RUNS.get(key, 0) + 1
    r = dict(_RUNS); r[key] = n
    for k in list(r)[:-40]:
        del r[k]
    _RUNS = r
    _persist()
    return n * 1_000_003


def _champion_key(opts):
    return (f"{opts['target']:g}|{opts['w']}x{opts['h']}|m{opts['mult']:g}"
            f"|t{opts['maxlvl']}|u{int(opts['uncap_pwr'])}|o{opts.get('objective', 'merges')}")


def _as_archive(v):
    if not v:
        return []
    return list(v) if isinstance(v[0], list) else [v]


def _priors(ctx, target, cap=12):
    scored = []
    for v in _CHAMPIONS.values():
        for grid in _as_archive(v):
            if len(grid) != ctx.n:
                continue
            if any(c and c[0] != 'CMP' and c[1] > ctx.type_max(c[0]) for c in grid):
                continue
            ev = evaluate(ctx, grid)
            if not (ev['feasible'] and ev['bits'] >= target - EPS):
                continue
            scored.append((ev[ctx.objective], grid))
    scored.sort(key=lambda x: x[0])
    seen = set(); out = []
    for _, g in scored:
        k = tuple(g)
        if k in seen:
            continue
        seen.add(k); out.append(g)
        if len(out) >= cap:
            break
    return out


def _save_champion(key, grid):
    global _CHAMPIONS
    d = dict(_CHAMPIONS)
    arch = _as_archive(d.get(key))
    arch.append(grid)
    seen = set(); uniq = []
    for g in reversed(arch):
        k = tuple(g)
        if k in seen:
            continue
        seen.add(k); uniq.append(g)
    d[key] = list(reversed(uniq))[-12:]
    for k in list(d)[:-20]:
        del d[k]
    _CHAMPIONS = d
    _persist()


def _parse_time(s):
    # Accepts bare seconds ("90"), single units ("20m", "1.5h", "45s", "3min"),
    # and combinations ("1h30m", "1m30s"). Bare numbers are seconds.
    import re
    s = str(s).strip().lower().replace('min', 'm').replace(' ', '')
    if not s:
        raise ValueError('empty')
    unit = {'h': 3600.0, 'm': 60.0, 's': 1.0, '': 1.0}
    tokens = re.findall(r'(\d*\.?\d+)([hms]?)', s)
    if not tokens or ''.join(n + u for n, u in tokens) != s:
        return float(s)  # malformed -> let float() raise for the caller to catch
    return sum(float(n) * unit[u] for n, u in tokens)


def run_gui():
    import threading, queue, time
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Grid Optimizer")
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=(22, 20))
    frm.grid()
    frm.columnconfigure(2, weight=1)
    _row = [0]
    def nextrow():
        _row[0] += 1
        return _row[0]

    INPUT_W = 12
    ROWPAD = 5
    LBLPAD = (0, 14)
    HINTPAD = (16, 0)
    HINT_FG = '#8a8a8a'

    ttk.Label(frm, text="Grid Optimizer", font=('Segoe UI', 14, 'bold')).grid(
        column=0, row=0, columnspan=3, pady=(0, 16), sticky='w')

    def _label(r, text):
        ttk.Label(frm, text=text).grid(column=0, row=r, sticky='w', padx=LBLPAD, pady=ROWPAD)

    def _hint(r, text):
        if text:
            ttk.Label(frm, text=text, foreground=HINT_FG).grid(
                column=2, row=r, sticky='w', padx=HINTPAD, pady=ROWPAD)

    def entry(label, default, hint=''):
        r = nextrow()
        _label(r, label)
        var = tk.StringVar(value=str(default))
        ttk.Entry(frm, textvariable=var, width=INPUT_W).grid(
            column=1, row=r, sticky='w', pady=ROWPAD)
        _hint(r, hint)
        return var

    def check(label, default):
        r = nextrow()
        var = tk.BooleanVar(value=bool(default))
        ttk.Checkbutton(frm, text=label, variable=var).grid(
            column=0, row=r, columnspan=3, sticky='w', pady=(2, 0))
        return var

    def combo(label, default, values, hint=''):
        r = nextrow()
        _label(r, label)
        var = tk.StringVar(value=default)
        ttk.Combobox(frm, textvariable=var, values=values, state='readonly',
                     width=INPUT_W + 6).grid(column=1, row=r, sticky='w', pady=ROWPAD)
        _hint(r, hint)
        return var

    OBJ_LABELS = [('merges', 'Fewest manual merges'), ('cost', 'Lowest token cost')]
    to_label = dict(OBJ_LABELS)
    to_obj = {v: k for k, v in OBJ_LABELS}

    S = _load_settings()
    tb = int(TARGET_BITS) if TARGET_BITS == int(TARGET_BITS) else TARGET_BITS
    v_target = entry("Target Bits/Sec", S.get('target', tb))
    v_w      = entry("Grid Width", S.get('w', 6))
    v_h      = entry("Grid Height", S.get('h', 6))
    v_time   = entry("Time Limit", S.get('time_str', S.get('time_limit', 180)),
                     hint="Seconds; add m/h or combine (e.g. 20m, 1h30m). Longer = better.")
    v_mult   = entry("Bits Multiplier", S.get('mult', 1.6), hint='Global multiplier on bits output.')
    v_maxlvl = entry("Max Tier", S.get('maxlvl', 10), hint='Deepest tier allowed; lower solves faster.')
    v_obj    = combo("Optimize For", to_label.get(S.get('objective', 'merges'), OBJ_LABELS[0][1]),
                     [lbl for _, lbl in OBJ_LABELS],
                     hint='Fewest merges = easiest to build; lowest cost = cheapest to buy.')
    v_uncap  = check("Uncap Power Supply tier (ignores Max Tier)", S.get('uncap_pwr', False))

    run_btn = ttk.Button(frm, text="Run")
    run_btn.grid(column=0, row=nextrow(), columnspan=3, pady=(18, 0), sticky='we')
    status = ttk.Label(frm, text="Ready.", foreground='#555', anchor='center')
    status.grid(column=0, row=nextrow(), columnspan=3, sticky='we', pady=(14, 6))
    BAR_H = 16
    bar = tk.Canvas(frm, height=BAR_H, highlightthickness=1,
                    highlightbackground='#c9c9c9', bg='#ededed')
    bar.grid(column=0, row=nextrow(), columnspan=3, sticky='we')
    _fill = bar.create_rectangle(0, 0, 0, BAR_H, fill='#22c55e', width=0)

    def set_progress(frac):
        bar.update_idletasks()
        w = bar.winfo_width()
        bar.coords(_fill, 0, 0, int(max(0.0, min(1.0, frac)) * w), BAR_H)

    prog = {'t0': None, 'limit': 1.0}
    stop_event = threading.Event()
    q = queue.Queue()

    def work(opts):
        try:
            cfg = Config(opts['objective'], opts['mult'], opts['maxlvl'],
                         opts['uncap_pwr'], opts['w'], opts['h'])
            ctx = Ctx(cfg)
            key = _champion_key(opts)
            _prefetch_ledger(ctx); _LEDGER_NEW.clear()
            tgt = opts['target']
            def on_improve(gr):
                ev = evaluate(ctx, gr)
                ok = ev['feasible'] and ev['bits'] >= tgt - EPS
                metric = (f"{ev['merges']} merges" if opts['objective'] == 'merges'
                          else f"{ev['cost']:.0f} cost")
                q.put(('progress', f"Working... best so far: {metric}, {ev['bits']:.0f} bits/s"
                                   f"{'' if ok else ' (below target)'}"))
            grid = solve_engine(ctx, cfg, tgt, opts['time_limit'],
                                priors=_priors(ctx, tgt), stop_event=stop_event,
                                seed_off=_next_seed_off(key), progress=on_improve)
            _save_champion(key, grid)
            _save_ledger()
            ev = evaluate(ctx, grid)
            write_and_open_html(ctx, grid, tgt)
            ok = ev['feasible'] and ev['bits'] >= tgt - EPS
            metric = (f"{ev['merges']} merges" if opts['objective'] == 'merges'
                      else f"{ev['cost']:.0f} token cost")
            note = "" if _ortools_available() else " (exact solver off: install 'ortools' for better results)"
            q.put(('done', f"{'Done' if ok else 'Best found (below target)'}: "
                           f"{ev['bits']:.0f} bits/s, {metric}. Opened in browser.{note}"))
        except Exception as e:
            q.put(('error', f"Error: {e}"))

    def poll():
        try:
            kind, msg = q.get_nowait()
        except queue.Empty:
            if prog['t0'] is not None:
                set_progress(min(0.99, (time.time() - prog['t0']) / prog['limit']))
            root.after(120, poll); return
        if kind == 'progress':
            status.config(text=msg, foreground='#555')
            root.after(120, poll); return
        set_progress(1.0 if kind == 'done' else 0.0)
        prog['t0'] = None
        status.config(text=msg, foreground='#b00' if kind == 'error' else '#0a0')
        run_btn.config(state='normal', text='Run', command=on_run)

    def on_stop():
        stop_event.set()
        run_btn.config(state='disabled', text='Stopping...')
        status.config(text="Stopping -- finishing the current best...", foreground='#555')

    def on_run():
        try:
            raw_time = v_time.get()
            opts = dict(
                target=float(v_target.get()),
                w=int(v_w.get()), h=int(v_h.get()),
                mult=float(v_mult.get()),
                maxlvl=int(v_maxlvl.get()),
                time_limit=_parse_time(raw_time),
                time_str=raw_time.strip(),
                uncap_pwr=bool(v_uncap.get()),
                objective=to_obj.get(v_obj.get(), 'merges'),
            )
            assert opts['w'] >= 1 and opts['h'] >= 1 and opts['target'] > 0
            assert opts['mult'] > 0 and 1 <= opts['maxlvl'] <= 20
            assert opts['time_limit'] >= 1
        except Exception:
            status.config(text="Check inputs: target>0, width/height>=1, "
                               "multiplier>0, max tier 1-20, time limit>=1.",
                          foreground='#b00')
            return
        _save_settings(opts)
        stop_event.clear()
        run_btn.config(text='Stop', command=on_stop)
        status.config(text=f"Solving {opts['w']}x{opts['h']} for >= "
                           f"{opts['target']:g} bits/sec (up to {opts['time_limit']:g}s)... "
                           f"Stop to end early and keep the best.", foreground='#555')
        prog['t0'] = time.time(); prog['limit'] = max(1.0, opts['time_limit'])
        set_progress(0.0)
        threading.Thread(target=work, args=(opts,), daemon=True).start()
        root.after(120, poll)

    run_btn.config(command=on_run)
    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 3
    root.geometry(f'{w}x{h}+{x}+{y}')
    root.mainloop()


def _relaunch_windowless():
    if os.name != 'nt':
        return False
    exe = sys.executable or ''
    if exe.lower().endswith('pythonw.exe'):
        return False
    pythonw = os.path.join(os.path.dirname(exe), 'pythonw.exe')
    if not os.path.exists(pythonw):
        return False
    import subprocess
    try:
        subprocess.Popen([pythonw, os.path.abspath(__file__)] + sys.argv[1:],
                         creationflags=0x08000000)
        return True
    except Exception:
        return False


def _sym_variants(gr, w, h):
    def at(r, c):
        return gr[r * w + c]
    out = [[at(r, w - 1 - c) for r in range(h) for c in range(w)],   # hflip
           [at(h - 1 - r, c) for r in range(h) for c in range(w)],   # vflip
           [at(h - 1 - r, w - 1 - c) for r in range(h) for c in range(w)]]  # 180
    if w == h:
        out.append([at(c, r) for r in range(h) for c in range(w)])   # transpose
    return out


def _selftest():
    rng = random.Random(0)
    fast_bad = 0; scan_bad = 0; checks = 0; canon_bad = 0; canon_checks = 0
    for (w, h) in [(5, 4), (6, 4), (6, 6)]:
        for objective in ('merges', 'cost'):
            ctx = Ctx(Config(objective, 1.6, 8, False, w, h))
            for _ in range(40):
                gr = [random_cell(ctx, rng, k) for k in range(ctx.n)]
                gr[rng.randrange(ctx.n)] = ('CMP', 1)
                if objective == 'merges':          # objective-independent checks: run once
                    ev = evaluate(ctx, gr); f, b, m, d = _fast_eval(ctx, gr)
                    if f != ev['feasible'] or abs(b - ev['bits']) > 1e-6 or m != ev['merges']:
                        fast_bad += 1
                    ck = _canon_key(ctx, gr)
                    for v in _sym_variants(gr, w, h):
                        canon_checks += 1
                        if _canon_key(ctx, v) != ck:
                            canon_bad += 1
                for i in range(ctx.n):
                    if gr[i] and gr[i][0] == 'CMP':
                        continue
                    bs, bo = None, gr[i]
                    for opt in [gr[i]] + [o for o in _cell_options(ctx, i) if o != gr[i]]:
                        tr = list(gr); tr[i] = opt
                        s, _ = score(ctx, tr, 800)
                        if bs is None or s < bs - 1e-9:
                            bs, bo = s, opt
                    io, _, _ = _scan_cell(ctx, gr, i, 800); checks += 1
                    if io != bo:
                        scan_bad += 1
    print(f"_fast_eval vs evaluate mismatches: {fast_bad}")
    print(f"_scan_cell vs score mismatches (merges+cost): {scan_bad}/{checks}")
    print(f"canon-key symmetry mismatches: {canon_bad}/{canon_checks}")
    ok = fast_bad == 0 and scan_bad == 0 and canon_bad == 0
    print("PASS" if ok else "FAIL")
    return ok


def main():
    _load_state()
    if len(sys.argv) >= 2 and sys.argv[1] == '--selftest':
        sys.exit(0 if _selftest() else 1)
    if len(sys.argv) >= 2 and sys.argv[1] == '--partest':
        import time
        maxlvl = int(sys.argv[4]) if len(sys.argv) > 4 else 8
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
        cfg = Config('merges', 1.6, maxlvl, False, 6, 6)
        ctx = Ctx(cfg)
        t = time.time()
        grid = solve_engine(ctx, cfg, 1220.0, steps)
        ev = evaluate(ctx, grid)
        print(f"solve_engine: merges={ev['merges']} feasible={ev['feasible']} bits={ev['bits']:.0f} "
              f"maxtier={maxlvl} in {time.time()-t:.0f}s")
        return
    if len(sys.argv) == 1:
        if _relaunch_windowless():
            return
        try:
            run_gui()
            return
        except Exception:
            pass
    if os.name == 'nt':
        os.system('')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    target = TARGET_BITS
    w, h = 5, 4
    if len(sys.argv) > 1:
        target = float(sys.argv[1])
    if len(sys.argv) > 3:
        w, h = int(sys.argv[2]), int(sys.argv[3])
    cfg = Config('merges', 1.6, 10, False, w, h)
    ctx = Ctx(cfg)
    print(f"Optimizing {w}x{h} grid for >= {target:g} bits/sec ...")
    grid = optimize(ctx, target)
    report(ctx, grid, target)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
