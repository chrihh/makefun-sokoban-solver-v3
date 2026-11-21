# Unified whole-board Sokoban solver (v3)

An entry for make.fun's "Build the best Sokoban solver, 21 November 2026"
bounty ($100 in USDC). This is a third, independently designed version of
the author's earlier entries: where the previous version decomposed the
board first and attacked each room with two engines, this one leads with
a single unified whole-board search and keeps decomposition as a
fallback.

`main.ts` (Node 22, no dependencies) reads each level at
`/input/<id>/level.txt` (or `$MAKEFUN_INPUT_DIR`) and writes its moves to
`/output/<id>/solution.txt` (or `$MAKEFUN_OUTPUT_DIR`).

How it works:

- **One A\* engine over pushes.** A state is the sorted box tuple plus
  the player; expansions are pushes, and the walks between pushes are
  flood-filled, never searched. The transposition table is keyed by a
  canonical player representative — the smallest square of the player's
  reachable region — so states that differ only in where the player
  stands inside one region merge. A second map prunes duplicate
  generations, and the heap is size-guarded, all by fixed counts.
- **Assignment heuristic.** For every goal, an exact reverse-pull search
  over (box, player) states marks the squares a lone box can never
  bring to that goal from, and a box-only pull BFS prices the rest in
  pushes. Boxes are matched to distinct goals at minimum total price
  with the **Hungarian algorithm** (boards of up to 10 boxes; larger
  boards use the admissible nearest-goal sum), memoised per box tuple.
  Pushes are generated in heuristic-delta order, tunnel squares winning
  ties.
- **Deadlock pruning.** A push onto a square outside every goal's exact
  reachable set is refused, as are states with a **frozen box** off its
  goal (fixed-point freeze detection: a box barred on one side of each
  axis by walls or already-frozen boxes never moves again) and dead
  2x2 blocks.
- **Fallback: dead-cut rooms.** If the unified pass exhausts its
  expansion budget, the exact reachable sets double as a scalpel:
  deleting every square a box can never use splits these generated
  boards into independent room components, each holding as many goals
  as boxes. Each room is solved by the same engine on a zoned sub-board
  — its boxes may not leave the room, while the player still walks the
  real floor (transit and dead squares included, other rooms' interiors
  excluded). Entry is by walking in when the doorway is open, or from
  the stand square outside it when a room box sits in the doorway, so
  the engine itself makes the push that opens the room. Rooms are
  stitched together with walks on the real board, and the assembled
  move string is replay-verified in full before it is used. A final
  large-budget unified pass covers boards the split does not recognise.
- Every candidate solution is **replayed on the real board before it is
  written**; all budgets count expanded states, never the clock, and
  every tie breaks by a fixed rule, so repeated runs write identical
  bytes.

Verified locally with the kit's own checker: on the practice set it
solves every level in exactly the optimal number of pushes
(13 / 16 / 29 / 47 / 60) for the maximum score of 1,000,000 in about
1.6 s, and on six freshly drawn generator sets (30 levels) it was
push-optimal on every level, byte-identical across double runs. The
house baseline plain-astar scores 200,000 on the same practice set.
