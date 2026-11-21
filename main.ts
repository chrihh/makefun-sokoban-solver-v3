// Unified whole-board push-optimal Sokoban solver (v3).
//
// Entry for make.fun's "Build the best Sokoban solver, 21 November 2026"
// bounty. Node 22, no dependencies. Reads each level at
// /input/<id>/level.txt (or $MAKEFUN_INPUT_DIR) and writes the player's
// moves to /output/<id>/solution.txt (or $MAKEFUN_OUTPUT_DIR).
//
// Architecture (one engine, two passes):
//
//   Pass 1 is a single unified A* over the WHOLE board at push level: a
//   state is the sorted box tuple plus the player, expansions are pushes,
//   and walks between pushes are flood-filled, never searched. The
//   transposition table is keyed by a canonical player representative —
//   the smallest cell of the player's reachable region — so states that
//   differ only in where the player stands inside one region merge.
//   Children are generated in heuristic-delta order (biggest estimated
//   improvement first, tunnels preferred on ties) so the heap receives
//   promising pushes early.
//
//   The heuristic is an assignment bound: for every goal, an exact
//   reverse-pull reachability search over (box, player) states marks the
//   squares a lone box can never bring to that goal from, and a cheaper
//   box-only pull BFS prices the rest in pushes. Boxes are matched to
//   distinct goals at minimum total price with the Hungarian algorithm
//   (boards of up to 10 boxes; larger boards use the admissible
//   nearest-goal sum), memoised per box tuple. A push that lands a box
//   on a square outside every goal's reachable set is pruned outright,
//   as are frozen boxes (fixed-point freeze detection) and dead 2x2
//   blocks.
//
//   Pass 2 runs only if pass 1 exhausts its expansion budget. The exact
//   pull-reachability sets double as a scalpel: deleting every square a
//   box can never use splits these generated boards into independent
//   room components, each holding as many goals as boxes. Each room is
//   then solved by the same A* engine on a zoned sub-board: its boxes
//   may not leave the room's cells, while the player still walks the
//   real floor (transit and dead squares included, other rooms'
//   interiors excluded). The player is walked between rooms on the
//   real board, and the assembled move string is replay-verified in full
//   before it is used. If the split is not recognised, pass 3 retries
//   the whole board with a much larger budget.
//
// Every budget counts expanded states, never the clock, and every tie
// breaks by a fixed rule, so repeated runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in a fixed order: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];
const INF = 1e9;

interface Board {
  w: number;
  h: number;
  n: number;
  wall: Uint8Array; // 1 = wall for the player (movement and standing)
  boxWall: Uint8Array; // 1 = a box may not enter (wall, or outside the room zone)
  goal: Uint8Array;
  goals: number[];
  boxes: number[]; // sorted
  player: number;
}

function parseBoard(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const n = w * h;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') { goal[i] = 1; goals.push(i); }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, n, wall, boxWall: wall, goal, goals, boxes, player };
}

// ---------------------------------------------------------------------------
// Per-goal pull analysis.
//
// exactReach(goal): the set of box squares from which a lone box can be
// pushed onto `goal`, computed exactly by reverse search over
// (box, player) states — reverse walks move the player, reverse pushes
// undo a push. Player movement is fully modelled, so a square absent
// from this set is a true dead square for that goal.
//
// pullDist(goal): an admissible push price for a lone box on each
// square, from a box-only pull BFS (the player is assumed able to stand
// wherever a pull requires; that only ever underprices).
// ---------------------------------------------------------------------------

function exactReach(b: Board, goalCell: number): Uint8Array {
  const { n, wall, w } = b;
  const reach = new Uint8Array(n);
  const seen = new Uint8Array(n * n);
  const queue: number[] = [];
  for (let p = 0; p < n; p++) {
    if (wall[p] || p === goalCell) continue;
    seen[goalCell * n + p] = 1;
    queue.push(goalCell * n + p);
  }
  reach[goalCell] = 1;
  let head = 0;
  while (head < queue.length) {
    const state = queue[head++];
    const box = (state / n) | 0;
    const player = state % n;
    reach[box] = 1;
    const bx = box % w, by = (box / w) | 0;
    const px = player % w, py = (player / w) | 0;
    // Reverse walk: the player could have stood on any neighbour square.
    for (let d = 0; d < 4; d++) {
      const qx = px + DX[d], qy = py + DY[d];
      if (qx < 0 || qy < 0 || qx >= w || qy >= b.h) continue;
      const q = qy * w + qx;
      if (wall[q] || q === box) continue;
      const s = box * n + q;
      if (!seen[s]) { seen[s] = 1; queue.push(s); }
    }
    // Reverse push: a forward push along d ends with box = player + d.
    for (let d = 0; d < 4; d++) {
      if (bx !== px + DX[d] || by !== py + DY[d]) continue;
      const prevBox = player; // the square the player now occupies
      const sx = px - DX[d], sy = py - DY[d];
      if (sx < 0 || sy < 0 || sx >= w || sy >= b.h) continue;
      const prevPlayer = sy * w + sx;
      if (wall[prevPlayer]) continue;
      const s = prevBox * n + prevPlayer;
      if (!seen[s]) { seen[s] = 1; queue.push(s); }
    }
  }
  return reach;
}

function pullDist(b: Board, goalCell: number): Int32Array {
  const { n, wall, w } = b;
  const dist = new Int32Array(n).fill(INF);
  dist[goalCell] = 0;
  const queue = [goalCell];
  let head = 0;
  while (head < queue.length) {
    const box = queue[head++];
    const bx = box % w, by = (box / w) | 0;
    for (let d = 0; d < 4; d++) {
      // Pull the box from (bx - DX, by - DY): that square must be free,
      // and the player must be able to stand one square further on.
      const cx = bx - DX[d], cy = by - DY[d];
      const sx = bx - 2 * DX[d], sy = by - 2 * DY[d];
      if (cx < 0 || cy < 0 || cx >= w || cy >= b.h) continue;
      if (sx < 0 || sy < 0 || sx >= w || sy >= b.h) continue;
      const cell = cy * w + cx;
      if (wall[cell] || wall[sy * w + sx] || dist[cell] !== INF) continue;
      dist[cell] = dist[box] + 1;
      queue.push(cell);
    }
  }
  return dist;
}

interface Tables {
  dead: Uint8Array; // 1 = a box here can never reach any goal
  cost: Int32Array[]; // cost[boxIndex?] no: costPerGoal[goalIdx][cell]
  reachUnion: Uint8Array;
}

function buildTables(b: Board): Tables {
  const reachUnion = new Uint8Array(b.n);
  const costPerGoal: Int32Array[] = [];
  for (const g of b.goals) {
    const r = exactReach(b, g);
    for (let i = 0; i < b.n; i++) if (r[i]) reachUnion[i] = 1;
    costPerGoal.push(pullDist(b, g));
  }
  const dead = new Uint8Array(b.n);
  for (let i = 0; i < b.n; i++) if (!b.wall[i] && !reachUnion[i]) dead[i] = 1;
  return { dead, cost: costPerGoal, reachUnion };
}

// ---------------------------------------------------------------------------
// Assignment heuristic (Hungarian algorithm, O(n^3), square cost matrix).
// Returns the minimum total price of matching boxes to distinct goals,
// or INF when some box cannot reach any goal it could be assigned.
// ---------------------------------------------------------------------------

function hungarian(cost: number[][]): number {
  const n = cost.length;
  const u = new Array(n + 1).fill(0);
  const v = new Array(n + 1).fill(0);
  const p = new Array(n + 1).fill(0); // p[j] = row matched to column j
  const way = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(n + 1).fill(Infinity);
    const used = new Array(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity, j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= n; j++) {
    if (p[j] === 0) return INF;
    const c = cost[p[j] - 1][j - 1];
    if (c >= INF) return INF;
    total += c;
  }
  return total;
}

// ---------------------------------------------------------------------------
// The A* engine. Works on any Board view (whole board or one room with
// the rest walled off). Push-level search; walks are flood-filled.
// ---------------------------------------------------------------------------

interface EngineResult { moves: string; pushes: number; expanded: number }

function solveWithAStar(b: Board, budget: number): EngineResult | null {
  const { w, n, wall, goal, goals } = b;
  const nBoxes = b.boxes.length;
  if (nBoxes === 0) return { moves: '', pushes: 0, expanded: 0 };
  const tables = buildTables(b);
  if (b.boxes.some((x) => tables.dead[x] && !goal[x])) return null;

  // Tunnel squares: floor with walls on both sides of one axis. A push
  // into a tunnel is a commitment, so tunnels win ordering ties.
  const tunnel = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (wall[i]) continue;
    const x = i % w, y = (i / w) | 0;
    const wl = x === 0 || wall[i - 1], wr = x === w - 1 || wall[i + 1];
    const wu = y === 0 || wall[i - w], wd = y === b.h - 1 || wall[i + w];
    if ((wl && wr) || (wu && wd)) tunnel[i] = 1;
  }

  // Heuristic, memoised per box tuple.
  const hCache = new Map<string, number>();
  const heuristic = (boxes: number[]): number => {
    const key = String.fromCharCode(...boxes);
    const hit = hCache.get(key);
    if (hit !== undefined) return hit;
    let h: number;
    if (nBoxes <= 10 && goals.length === nBoxes) {
      const matrix: number[][] = boxes.map((box) =>
        goals.map((_, gi) => tables.cost[gi][box]));
      h = hungarian(matrix);
    } else {
      h = 0;
      for (const box of boxes) {
        let best = INF;
        for (let gi = 0; gi < goals.length; gi++) {
          const c = tables.cost[gi][box];
          if (c < best) best = c;
        }
        if (best >= INF) { h = INF; break; }
        h += best;
      }
    }
    hCache.set(key, h);
    return h;
  };

  const boxSetOf = (boxes: number[]): Set<number> => new Set(boxes);

  // Fixed-point freeze detection. A box that cannot move along either
  // axis — because a wall or an already-frozen box bars one side of
  // each axis — never moves again; if it is off its goal the state is
  // dead. Sound because frozen boxes only ever act as extra walls.
  const freezeKills = (boxes: number[], boxSet: Set<number>): boolean => {
    const frozen = new Set<number>();
    const blocked = (i: number): boolean => wall[i] === 1 || frozen.has(i);
    let changed = true;
    while (changed) {
      changed = false;
      for (const box of boxes) {
        if (frozen.has(box)) continue;
        const x = box % w;
        const hl = x === 0 || blocked(box - 1), hr = x === w - 1 || blocked(box + 1);
        const vu = blocked(box - w), vd = blocked(box + w);
        if ((hl || hr) && (vu || vd)) { frozen.add(box); changed = true; }
      }
    }
    for (const box of frozen) if (!goal[box]) return true;
    return false;
  };

  // Dead 2x2 block check around a freshly pushed box.
  const blockKills = (dest: number, boxSet: Set<number>): boolean => {
    const solid = (i: number): boolean => wall[i] === 1 || boxSet.has(i);
    const dx = dest % w, dy = (dest / w) | 0;
    for (const ox of [-1, 0]) for (const oy of [-1, 0]) {
      const x0 = dx + ox, y0 = dy + oy;
      if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= b.h) continue;
      const cells = [y0 * w + x0, y0 * w + x0 + 1, (y0 + 1) * w + x0, (y0 + 1) * w + x0 + 1];
      if (cells.every(solid) && cells.some((c) => boxSet.has(c) && !goal[c])) return true;
    }
    return false;
  };

  // Node store.
  const nodeBoxes: number[][] = [b.boxes.slice()];
  const nodePlayer: number[] = [b.player];
  const nodeG: number[] = [0];
  const nodeParent: number[] = [-1];
  const nodeSeg: string[] = [''];
  const h0 = heuristic(b.boxes);
  if (h0 >= INF) return null;

  // Binary heap of [f, h, seq, nodeId], ordered by f, then h, then seq.
  const heap: number[][] = [];
  let seq = 0;
  const less = (a: number[], c: number[]): boolean =>
    a[0] < c[0] || (a[0] === c[0] && (a[1] < c[1] || (a[1] === c[1] && a[2] < c[2])));
  const heapPush = (e: number[]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = (): number[] => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  heapPush([h0, h0, seq++, 0]);

  // generatedBest prunes duplicate (actual player, boxes) generations;
  // expandedBest is the canonical transposition table (region, boxes).
  const generatedBest = new Map<string, number>();
  generatedBest.set(String.fromCharCode(b.player, ...b.boxes), 0);
  const expandedBest = new Map<string, number>();

  const prev = new Int32Array(n);
  const queue = new Int32Array(n);
  let expanded = 0;

  while (heap.length > 0) {
    const [, , , id] = heapPop();
    const boxes = nodeBoxes[id];
    const player = nodePlayer[id];
    const g = nodeG[id];

    // Flood fill the player's region; canonical representative = its
    // smallest cell. Also leaves `prev` ready for walk reconstruction.
    prev.fill(-2);
    let qh = 0, qt = 0;
    queue[qt++] = player;
    prev[player] = -1;
    let canon = player;
    const boxSet = boxSetOf(boxes);
    while (qh < qt) {
      const c = queue[qh++];
      if (c < canon) canon = c;
      const cx = c % w, cy = (c / w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d], ny = cy + DY[d];
        if (nx < 0 || ny < 0 || nx >= w || ny >= b.h) continue;
        const nc = ny * w + nx;
        if (prev[nc] !== -2 || wall[nc] || boxSet.has(nc)) continue;
        prev[nc] = c;
        queue[qt++] = nc;
      }
    }
    const canonKey = String.fromCharCode(canon, ...boxes);
    const seenG = expandedBest.get(canonKey);
    if (seenG !== undefined && seenG <= g) continue;
    expandedBest.set(canonKey, g);

    if (boxes.every((x) => goal[x] === 1)) {
      let moves = '';
      for (let t = id; t > 0; t = nodeParent[t]) moves = nodeSeg[t] + moves;
      return { moves, pushes: g, expanded };
    }
    if (expanded >= budget) return null;
    if (heap.length > 1_500_000) return null; // deterministic size guard
    expanded++;

    // Generate pushes, ordered by heuristic delta (best first), tunnel
    // destinations winning ties, then by box and direction — all fixed
    // rules, so generation order is deterministic.
    interface Child { boxes: number[]; player: number; seg: string; h: number; tun: number; boxPos: number; dir: number }
    const children: Child[] = [];
    for (let bi = 0; bi < boxes.length; bi++) {
      const box = boxes[bi];
      const bx = box % w, by = (box / w) | 0;
      for (let d = 0; d < 4; d++) {
        const sx = bx - DX[d], sy = by - DY[d]; // player stand square
        const tx = bx + DX[d], ty = by + DY[d]; // box destination
        if (sx < 0 || sy < 0 || sx >= w || sy >= b.h) continue;
        if (tx < 0 || ty < 0 || tx >= w || ty >= b.h) continue;
        const stand = sy * w + sx, dest = ty * w + tx;
        if (prev[stand] === -2) continue; // player cannot reach the stand square
        if (b.boxWall[dest] || boxSet.has(dest)) continue;
        if (tables.dead[dest] && !goal[dest]) continue;
        const nb = boxes.slice();
        nb[bi] = dest;
        nb.sort((a2, b2) => a2 - b2);
        const nSet = boxSetOf(nb);
        if (blockKills(dest, nSet)) continue;
        // Freeze check only when the pushed box touches something on
        // both axes — the only way this push can freeze anything.
        const touchesH = (bx > 0 && (wall[box - 1] || nSet.has(box - 1))) ||
          (bx < w - 1 && (wall[box + 1] || nSet.has(box + 1))) ||
          (tx > 0 && (wall[dest - 1] || nSet.has(dest - 1))) ||
          (tx < w - 1 && (wall[dest + 1] || nSet.has(dest + 1)));
        const touchesV = (wall[box - w] || nSet.has(box - w)) ||
          (wall[box + w] || nSet.has(box + w)) ||
          (wall[dest - w] || nSet.has(dest - w)) ||
          (wall[dest + w] || nSet.has(dest + w));
        if (touchesH && touchesV && freezeKills(nb, nSet)) continue;
        const h = heuristic(nb);
        if (h >= INF) continue;
        // Walk string from the player to the stand square.
        let walk = '';
        for (let c = stand; c !== player; ) {
          const p = prev[c];
          const dd = c - p;
          walk = (dd === 1 ? 'r' : dd === -1 ? 'l' : dd === w ? 'd' : 'u') + walk;
          c = p;
        }
        children.push({
          boxes: nb, player: box, seg: walk + LETTER[d].toUpperCase(),
          h, tun: tunnel[dest], boxPos: box, dir: d,
        });
      }
    }
    children.sort((a, c) =>
      (a.h - c.h) || (c.tun - a.tun) || (a.boxPos - c.boxPos) || (a.dir - c.dir));
    for (const child of children) {
      const gKey = String.fromCharCode(child.player, ...child.boxes);
      const prevBest = generatedBest.get(gKey);
      if (prevBest !== undefined && prevBest <= g + 1) continue;
      generatedBest.set(gKey, g + 1);
      const nid = nodeBoxes.length;
      nodeBoxes.push(child.boxes);
      nodePlayer.push(child.player);
      nodeG.push(g + 1);
      nodeParent.push(id);
      nodeSeg.push(child.seg);
      heapPush([g + 1 + child.h, child.h, seq++, nid]);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Replay verification on the real board.
// ---------------------------------------------------------------------------

function replay(b: Board, moves: string): { ok: boolean; solved: boolean; pushes: number } {
  const { w, wall, goal } = b;
  const boxes = new Set(b.boxes);
  let player = b.player;
  let pushes = 0;
  for (const ch of moves) {
    const d = LETTER.indexOf(ch.toLowerCase());
    if (d < 0) continue;
    const delta = DY[d] * w + DX[d];
    const next = player + delta;
    if (wall[next]) return { ok: false, solved: false, pushes };
    if (boxes.has(next)) {
      const beyond = next + delta;
      if (wall[beyond] || boxes.has(beyond)) return { ok: false, solved: false, pushes };
      boxes.delete(next);
      boxes.add(beyond);
      pushes++;
    }
    player = next;
  }
  return { ok: true, solved: [...boxes].every((x) => goal[x] === 1), pushes };
}

// ---------------------------------------------------------------------------
// Pass 2: dead-cut room decomposition. Squares outside the exact
// pull-reachability union are deleted; if the remaining floor splits
// into components that each hold as many goals as boxes, every room is
// an independent sub-board (a box can never productively leave its
// room), solved by the same engine and stitched together with walks.
// ---------------------------------------------------------------------------

function solveByRooms(b: Board): string | null {
  const tables = buildTables(b);
  const { n, wall } = b;
  const keep = new Uint8Array(n);
  for (let i = 0; i < n; i++) keep[i] = !wall[i] && tables.reachUnion[i] ? 1 : 0;
  const compId = new Int32Array(n).fill(-1);
  const comps: number[][] = [];
  for (let s = 0; s < n; s++) {
    if (!keep[s] || compId[s] >= 0) continue;
    const cells: number[] = [];
    const queue = [s];
    compId[s] = comps.length;
    while (queue.length > 0) {
      const c = queue.pop()!;
      cells.push(c);
      const cx = c % b.w, cy = (c / b.w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d], ny = cy + DY[d];
        if (nx < 0 || ny < 0 || nx >= b.w || ny >= b.h) continue;
        const nc = ny * b.w + nx;
        if (!keep[nc] || compId[nc] >= 0) continue;
        compId[nc] = comps.length;
        queue.push(nc);
      }
    }
    comps.push(cells);
  }
  const work = comps
    .map((cells) => ({
      cells,
      boxes: b.boxes.filter((x) => cells.includes(x)),
      goals: b.goals.filter((x) => cells.includes(x)),
    }))
    .filter((c) => c.boxes.length > 0);
  if (work.length === 0) return null;
  if (work.some((c) => c.boxes.length !== c.goals.length)) return null;
  if (work.reduce((a, c) => a + c.boxes.length, 0) !== b.boxes.length) return null;
  // A single component that already contains the player is just the
  // whole board minus dead squares — pass 1/pass 3 territory.
  if (work.length === 1 && work[0].cells.includes(b.player)) return null;
  work.sort((a, c) => Math.min(...a.cells) - Math.min(...c.cells));

  const boxesNow = new Set(b.boxes);
  let player = b.player;
  let allMoves = '';
  for (const room of work) {
    // Player floor for this room's sub-board: the original floor minus
    // the other rooms' interiors. Its boxes are zoned to the room.
    const subWall = b.wall.slice();
    for (const other of work) {
      if (other === room) continue;
      for (const c of other.cells) subWall[c] = 1;
    }
    const subBoxWall = new Uint8Array(n).fill(1);
    for (const c of room.cells) subBoxWall[c] = 0;

    // Where can the player walk right now, on the real board?
    const prev = new Int32Array(n).fill(-2);
    const rq = [player];
    prev[player] = -1;
    for (let head = 0; head < rq.length; head++) {
      const c = rq[head];
      const cx = c % b.w, cy = (c / b.w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d], ny = cy + DY[d];
        if (nx < 0 || ny < 0 || nx >= b.w || ny >= b.h) continue;
        const nc = ny * b.w + nx;
        if (prev[nc] !== -2 || b.wall[nc] || boxesNow.has(nc)) continue;
        prev[nc] = c;
        rq.push(nc);
      }
    }
    // Entry candidates: room cells the player can walk into, plus
    // stand squares for a push that moves a room box into the room
    // (the doorway may itself be blocked by a room box).
    const roomSet = new Set(room.cells);
    const cand = new Set<number>();
    for (const c of room.cells) if (prev[c] !== -2) cand.add(c);
    for (const box of room.boxes) {
      const bx = box % b.w, by = (box / b.w) | 0;
      for (let d = 0; d < 4; d++) {
        const sx = bx - DX[d], sy = by - DY[d];
        const tx = bx + DX[d], ty = by + DY[d];
        if (sx < 0 || sy < 0 || sx >= b.w || sy >= b.h) continue;
        if (tx < 0 || ty < 0 || tx >= b.w || ty >= b.h) continue;
        const stand = sy * b.w + sx, dest = ty * b.w + tx;
        if (!roomSet.has(dest) || boxesNow.has(dest) || b.wall[dest]) continue;
        if (prev[stand] !== -2) cand.add(stand);
      }
    }
    // One candidate per distinct player region of the sub-board.
    const roomBoxSet = new Set(room.boxes);
    const triedRegions = new Set<number>();
    let roomDone = false;
    for (const start of [...cand].sort((a, c) => a - c)) {
      // Region of `start` on the sub-board with the room's boxes fixed.
      const seen = new Uint8Array(n);
      const fq = [start];
      seen[start] = 1;
      let regionMin = start;
      while (fq.length > 0) {
        const c = fq.pop()!;
        if (c < regionMin) regionMin = c;
        const cx = c % b.w, cy = (c / b.w) | 0;
        for (let d = 0; d < 4; d++) {
          const nx = cx + DX[d], ny = cy + DY[d];
          if (nx < 0 || ny < 0 || nx >= b.w || ny >= b.h) continue;
          const nc = ny * b.w + nx;
          if (seen[nc] || subWall[nc] || roomBoxSet.has(nc)) continue;
          seen[nc] = 1;
          fq.push(nc);
        }
      }
      if (triedRegions.has(regionMin)) continue;
      triedRegions.add(regionMin);
      // Walk string from the player to this entry cell.
      let walk = '';
      for (let c = start; c !== player; ) {
        const p = prev[c];
        const dd = c - p;
        walk = (dd === 1 ? 'r' : dd === -1 ? 'l' : dd === b.w ? 'd' : 'u') + walk;
        c = p;
      }
      const sub: Board = {
        w: b.w, h: b.h, n,
        wall: subWall,
        boxWall: subBoxWall,
        goal: b.goal,
        goals: room.goals.slice(),
        boxes: room.boxes.slice().sort((a, c) => a - c),
        player: start,
      };
      const solved = solveWithAStar(sub, 2_000_000);
      if (!solved) continue;
      const segment = walk + solved.moves;
      const sim: Board = { ...b, boxes: [...boxesNow].sort((a, c) => a - c), player };
      const after = replayPartial(sim, segment);
      if (!after) continue;
      boxesNow.clear();
      for (const x of after.boxes) boxesNow.add(x);
      player = after.player;
      allMoves += segment;
      roomDone = true;
      break;
    }
    if (!roomDone) return null;
  }
  const check = replay(b, allMoves);
  return check.ok && check.solved ? allMoves : null;
}

// Replay that returns the end state instead of a verdict.
function replayPartial(b: Board, moves: string): { boxes: number[]; player: number } | null {
  const { w, wall } = b;
  const boxes = new Set(b.boxes);
  let player = b.player;
  for (const ch of moves) {
    const d = LETTER.indexOf(ch.toLowerCase());
    if (d < 0) continue;
    const delta = DY[d] * w + DX[d];
    const next = player + delta;
    if (wall[next]) return null;
    if (boxes.has(next)) {
      const beyond = next + delta;
      if (wall[beyond] || boxes.has(beyond)) return null;
      boxes.delete(next);
      boxes.add(beyond);
    }
    player = next;
  }
  return { boxes: [...boxes], player };
}

// ---------------------------------------------------------------------------
// Top level: pass 1 (unified, modest budget), pass 2 (rooms), pass 3
// (unified, large budget). First verified solution wins.
// ---------------------------------------------------------------------------

function solve(text: string): string {
  const b = parseBoard(text);
  if (b.boxes.every((x) => b.goal[x] === 1)) return '';
  const primary = solveWithAStar(b, b.boxes.length <= 8 ? 150_000 : 8_000);
  if (primary) {
    const check = replay(b, primary.moves);
    if (check.ok && check.solved) return primary.moves;
  }
  const byRooms = solveByRooms(b);
  if (byRooms !== null) return byRooms;
  const last = solveWithAStar(b, 600_000);
  if (last) {
    const check = replay(b, last.moves);
    if (check.ok && check.solved) return last.moves;
  }
  return '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
