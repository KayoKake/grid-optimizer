
import sys
import os
import math
import random
from collections import namedtuple, Counter

Config = namedtuple('Config', 'objective bits_mult max_level uncap_pwr w h')

TARGET_BITS = 78.0

LVL_OUT     = 1.20
LVL_WEIGHT  = 2.0
LVL_COST_SCALE = 1.30
PWR_ABS_MAX = 20
SCALE_CONS_WITH_LEVEL = True

OVR_BASE    = 1.00
OVR_PER_TIER = 0.20

def ovr_strength(lvl):
    return OVR_BASE + OVR_PER_TIER * (lvl - 1)

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

INFEASIBLE_PEN = 1e12
DEFICIT_W = 1e3
BELOW_TARGET_PEN = 1e8
SHORTFALL_W = 1e4


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
            s = ovr_strength(lvl)
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
        out_mult  = LVL_OUT ** (lvl - 1)
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
        out_mult = LVL_OUT ** (lvl - 1)
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
    cell = g[n]
    d = {r: 0.0 for r in RESOURCES}
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
            ovr += ovr_strength(o[1])
        if o[0] == t:
            same += 1
    out_mult  = LVL_OUT ** (lvl - 1)
    cons_mult = out_mult if SCALE_CONS_WITH_LEVEL else 1.0
    boost = 1.0 + ovr + 0.10 * same
    for r, amt in MODULES[t]['prod'].items():
        d[r] += amt * out_mult * boost
    for r, amt in MODULES[t]['cons'].items():
        d[r] -= amt * cons_mult
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


def _scan_cell(ctx, g, i, target):
    S = [i] + ctx.nb[i]
    ev0 = evaluate(ctx, g)
    bg = {r: ev0['totals'][r] for r in RESOURCES}
    bg['bits'] /= ctx.mult
    for n in S:
        c = _cell_contrib(ctx, g, n)
        for r in RESOURCES:
            bg[r] -= c[r]
    old = g[i]
    bg_cost = ev0['cost'] - _cell_cost(old)
    bg_merges = ev0['merges'] - _cell_merges(old)
    bg_misplaced = ev0['misplaced'] - _cell_misplaced(ctx, old, i)

    best_opt, best_s = old, None
    for opt in [old] + [o for o in _cell_options(ctx, i) if o != old]:
        g[i] = opt
        totals = {r: bg[r] for r in RESOURCES}
        for n in S:
            c = _cell_contrib(ctx, g, n)
            for r in RESOURCES:
                totals[r] += c[r]
        totals['bits'] *= ctx.mult
        cost = bg_cost + _cell_cost(opt)
        merges = bg_merges + _cell_merges(opt)
        obj = merges if ctx.objective == 'merges' else cost
        misplaced = bg_misplaced + _cell_misplaced(ctx, opt, i)
        feasible = misplaced == 0 and all(totals[r] >= -EPS for r in BALANCE)
        if not feasible:
            deficit = sum(-totals[r] for r in BALANCE if totals[r] < 0)
            s = INFEASIBLE_PEN + deficit * DEFICIT_W + obj
        elif totals['bits'] < target - EPS:
            s = BELOW_TARGET_PEN + (target - totals['bits']) * SHORTFALL_W + obj
        else:
            s = obj
        if best_s is None or s < best_s - 1e-9:
            best_s, best_opt = s, opt
    g[i] = old
    return best_opt, best_s


def polish(ctx, grid, target):
    n = ctx.n; corners = ctx.corners
    g = list(grid)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if g[i] and g[i][0] == 'CMP':
                continue
            best_opt, _ = _scan_cell(ctx, g, i, target)
            if best_opt != g[i]:
                g[i] = best_opt
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


def memetic(ctx, target, base_seed=1234, pop_size=24, gens=400, iters=3000, incumbent=None):
    rng = random.Random(base_seed)
    n = ctx.n
    corners = sorted(ctx.corners)
    pop = []
    if incumbent is not None:
        g = polish(ctx, list(incumbent), target)
        pop.append((score(ctx, g, target)[0], g))
    for r in range(pop_size):
        s = [random_cell(ctx, rng, k) for k in range(n)]
        s[corners[r] if r < len(corners) else rng.randrange(n)] = ('CMP', 1)
        g = polish(ctx, anneal(ctx, target, s, rng, iters=iters)[0], target)
        pop.append((score(ctx, g, target)[0], g))
    pop.sort(key=lambda x: x[0])
    for _ in range(gens):
        pa = min(rng.sample(pop, 3), key=lambda x: x[0])[1]
        pb = min(rng.sample(pop, 3), key=lambda x: x[0])[1]
        child = polish(ctx, _perturb(ctx, _crossover(ctx, pa, pb, rng), rng, rng.choice([0, 1, 2])), target)
        cs = score(ctx, child, target)[0]
        if cs < pop[-1][0]:
            pop[-1] = (cs, child)
            pop.sort(key=lambda x: x[0])
    return pop[0][1]


def optimize(ctx, target, restarts=150, iters=6000, base_seed=1234, incumbent=None,
             ils_iters=400, t0=8000.0):
    n = ctx.n
    best, best_s = None, float('inf')
    if incumbent is not None:
        g = polish(ctx, list(incumbent), target)
        best, best_s = g, score(ctx, g, target)[0]
    corners = sorted(ctx.corners)
    for r in range(restarts):
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


_SAVED = {'target': 1220.0, 'w': 6, 'h': 6, 'mult': 1.6, 'maxlvl': 10, 'time_limit': 180, 'uncap_pwr': False}
_CHAMPIONS = {}


def _cpsat_optimize(ctx, target, time_limit, hint=None, stop_event=None):
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None
    SO = 1000; SS = 1000
    N = ctx.n; nb = ctx.nb; corners = ctx.corners
    PLACE = ['PWR', 'CRW', 'RAM', 'DB', 'OVR', 'SAT']
    Lmax = max(ctx.type_max(t) for t in PLACE)
    OUT = [round(SO * (LVL_OUT ** (l - 1))) for l in range(1, Lmax + 1)]
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

    if hint is not None:
        for i in range(N):
            cell = hint[i]
            st = 'E' if not cell else ('C' if _SPECIAL[cell[0]] == 'compiler' else cell)
            if st in x[i]:
                mdl.AddHint(x[i][st], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = max(1, os.cpu_count() or 1)
    if stop_event is not None:
        class _Stopper(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                if stop_event.is_set():
                    self.StopSearch()
        status = solver.Solve(mdl, _Stopper())
    else:
        status = solver.Solve(mdl)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    grid = [None] * N
    for i in range(N):
        for s, v in x[i].items():
            if solver.Value(v):
                grid[i] = ('CMP', 1) if s == 'C' else (None if s == 'E' else s)
    return grid


def _worker(p):
    ctx = Ctx(p['cfg'])
    gens = max(80, p['restarts'] * 6)
    grid = memetic(ctx, p['target'], base_seed=p['seed'], pop_size=24,
                   gens=gens, iters=p['iters'])
    return (score(ctx, grid, p['target'])[0], grid)


def optimize_parallel(ctx, cfg, target, restarts, iters):
    if getattr(sys, 'frozen', False):
        return optimize(ctx, target, restarts=min(restarts, 8), iters=iters)
    try:
        import multiprocessing as mp
        ncores = os.cpu_count() or 1
    except Exception:
        ncores = 1
    ncores = max(1, min(ncores, max(1, restarts)))
    if ncores == 1:
        return optimize(ctx, target, restarts=restarts, iters=iters)
    base, extra = divmod(restarts, ncores)
    payloads = []
    for k in range(ncores):
        rs = base + (1 if k < extra else 0)
        if rs <= 0:
            continue
        payloads.append(dict(target=target, cfg=cfg,
                             restarts=rs, iters=iters, seed=1234 + k * 100003))
    try:
        mpctx = mp.get_context('spawn')
        with mpctx.Pool(len(payloads)) as pool:
            results = [r for r in pool.map(_worker, payloads) if r]
        if not results:
            return optimize(ctx, target, restarts=restarts, iters=iters)
        return min(results, key=lambda r: r[0])[1]
    except Exception:
        return optimize(ctx, target, restarts=restarts, iters=iters)


def solve_engine(ctx, cfg, target, time_limit, prior=None, stop_event=None):
    import time as _time
    start = _time.time()
    def rank(g):
        ev = evaluate(ctx, g)
        feasible = ev['feasible'] and ev['bits'] >= target - EPS
        return (0 if feasible else 1, ev['merges'])
    def stopped():
        return stop_event is not None and stop_event.is_set()
    pool = [g for g in (optimize_parallel(ctx, cfg, target, 48, 3000), prior) if g is not None]
    best = min(pool, key=rank) if pool else None
    if not stopped():
        cp = _cpsat_optimize(ctx, target, time_limit, hint=best, stop_event=stop_event)
        if cp is not None and (best is None or rank(cp) < rank(best)):
            best = cp
    while (_time.time() - start) < time_limit and not stopped():
        cand = optimize(ctx, target, restarts=6, iters=3000, incumbent=best)
        if best is None or rank(cand) < rank(best):
            best = cand
    return best


_FROZEN = getattr(sys, 'frozen', False)


def _load_settings():
    return dict(_SAVED)


def _state_path():
    base = os.path.dirname(sys.executable) if _FROZEN else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'grid_optimizer_state.txt')


def _load_state():
    global _SAVED, _CHAMPIONS
    try:
        import ast
        with open(_state_path(), 'r', encoding='utf-8') as f:
            st = ast.literal_eval(f.read())
        _SAVED = st.get('saved', _SAVED)
        _CHAMPIONS = st.get('champions', _CHAMPIONS)
    except Exception:
        pass


def _rewrite_self(name, value):
    try:
        import re
        path = os.path.abspath(__file__)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        new = re.sub(r'(?m)^%s = .*$' % name, '%s = %r' % (name, value), text, count=1)
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(new)
    except Exception:
        pass


def _persist():
    if _FROZEN:
        try:
            with open(_state_path(), 'w', encoding='utf-8') as f:
                f.write(repr({'saved': _SAVED, 'champions': _CHAMPIONS}))
        except Exception:
            pass
    else:
        _rewrite_self('_SAVED', _SAVED)
        _rewrite_self('_CHAMPIONS', _CHAMPIONS)


def _save_settings(d):
    global _SAVED
    _SAVED = dict(d)
    _persist()


def _champion_key(opts):
    return (f"{opts['target']:g}|{opts['w']}x{opts['h']}|m{opts['mult']:g}"
            f"|t{opts['maxlvl']}|u{int(opts['uncap_pwr'])}")


def _load_champion(key):
    return _CHAMPIONS.get(key)


def _best_prior(ctx, target):
    best = None; best_m = None
    for grid in _CHAMPIONS.values():
        if len(grid) != ctx.n:
            continue
        if any(c and c[0] != 'CMP' and c[1] > ctx.type_max(c[0]) for c in grid):
            continue
        ev = evaluate(ctx, grid)
        if not (ev['feasible'] and ev['bits'] >= target - EPS):
            continue
        if best is None or ev['merges'] < best_m:
            best_m, best = ev['merges'], grid
    return best


def _save_champion(key, grid):
    global _CHAMPIONS
    d = dict(_CHAMPIONS)
    d[key] = grid
    for k in list(d)[:-20]:
        del d[k]
    _CHAMPIONS = d
    _persist()


def _parse_time(s):
    s = str(s).strip().lower()
    mult = 1.0
    if s.endswith('min'):
        s = s[:-3]; mult = 60.0
    elif s.endswith('m'):
        s = s[:-1]; mult = 60.0
    elif s.endswith('s'):
        s = s[:-1]
    return float(s) * mult


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

    S = _load_settings()
    tb = int(TARGET_BITS) if TARGET_BITS == int(TARGET_BITS) else TARGET_BITS
    v_target = entry("Target Bits/Sec", S.get('target', tb))
    v_w      = entry("Grid Width", S.get('w', 6))
    v_h      = entry("Grid Height", S.get('h', 6))
    v_time   = entry("Time Limit", S.get('time_str', S.get('time_limit', 180)),
                     hint="Seconds, or add 'm' for minutes (e.g. 20m). Longer = better.")
    v_mult   = entry("Bits Multiplier", S.get('mult', 1.6), hint='Global multiplier on bits output.')
    v_maxlvl = entry("Max Tier", S.get('maxlvl', 10), hint='Deepest tier allowed; lower solves faster.')
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
            cfg = Config('merges', opts['mult'], opts['maxlvl'],
                         opts['uncap_pwr'], opts['w'], opts['h'])
            ctx = Ctx(cfg)
            key = _champion_key(opts)
            grid = solve_engine(ctx, cfg, opts['target'], opts['time_limit'],
                                prior=_best_prior(ctx, opts['target']),
                                stop_event=stop_event)
            _save_champion(key, grid)
            ev = evaluate(ctx, grid)
            write_and_open_html(ctx, grid, opts['target'])
            ok = ev['feasible'] and ev['bits'] >= opts['target'] - EPS
            q.put(('done', f"{'Done' if ok else 'Best found (below target)'}: "
                           f"{ev['bits']:.0f} bits/s, {ev['merges']} merges. Opened in browser."))
        except Exception as e:
            q.put(('error', f"Error: {e}"))

    def poll():
        try:
            kind, msg = q.get_nowait()
        except queue.Empty:
            if prog['t0'] is not None:
                set_progress(min(0.99, (time.time() - prog['t0']) / prog['limit']))
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


def _selftest():
    rng = random.Random(0)
    fast_bad = 0; scan_bad = 0; checks = 0
    for (w, h) in [(5, 4), (6, 4), (6, 6)]:
        ctx = Ctx(Config('merges', 1.6, 8, False, w, h))
        for _ in range(40):
            gr = [random_cell(ctx, rng, k) for k in range(ctx.n)]
            gr[rng.randrange(ctx.n)] = ('CMP', 1)
            ev = evaluate(ctx, gr); f, b, m, d = _fast_eval(ctx, gr)
            if f != ev['feasible'] or abs(b - ev['bits']) > 1e-6 or m != ev['merges']:
                fast_bad += 1
            for i in range(ctx.n):
                if gr[i] and gr[i][0] == 'CMP':
                    continue
                bs, bo = None, gr[i]
                for opt in [gr[i]] + [o for o in _cell_options(ctx, i) if o != gr[i]]:
                    tr = list(gr); tr[i] = opt
                    s, _ = score(ctx, tr, 800)
                    if bs is None or s < bs - 1e-9:
                        bs, bo = s, opt
                io, _ = _scan_cell(ctx, gr, i, 800); checks += 1
                if io != bo:
                    scan_bad += 1
    print(f"_fast_eval vs evaluate mismatches: {fast_bad}")
    print(f"_scan_cell vs score mismatches: {scan_bad}/{checks}")
    ok = fast_bad == 0 and scan_bad == 0
    print("PASS" if ok else "FAIL")
    return ok


def main():
    if _FROZEN:
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
