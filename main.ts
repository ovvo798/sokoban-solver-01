// Sokoban solver: push-optimal search per room, stitched with hallway walks.
//
// Idea (derived from the level structure: a 1-wide hallway with 1-4 rooms off
// it, boxes never leaving their room, a level's optimum being the sum of its
// rooms' optima): split the board into rooms, solve each room on its own with
// a breadth-first search over pushes (so the first solution found for a room
// uses the fewest pushes possible), then walk the player between rooms.
// A state is the boxes' squares plus the player's reachable region; two
// prunings keep the search small: a box may only ever sit on a "live" square
// (one from which pulls can drag it back onto a goal) and a box pushed into a
// frozen 2x2 block of walls and boxes (with a box off its goal) is dead.
// Everything is deterministic: fixed direction order, no clock, no random
// choices, work-bounded search caps.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in fixed order: [dx, dy, step letter]. Pushes use the same letter.
const DIRS: Array<[number, number, string]> = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[];
  boxes: number[]; // sorted
  player: number;
}

interface Solved {
  pushes: number;
  moves: string;
  endPlayer: number;
}

/** Parse XSB text into a level. Short rows and the outer frame count as walls. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * Live squares: squares a box can stand on while still being able to reach a
 * goal. Computed backwards from the goals: a square y is live when some live
 * square x can be reached by pulling a box from y to x (the player must be
 * able to stand at y - d, the square behind the box).
 */
function liveSquares(w: number, wall: Uint8Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(wall.length);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= wall.length) continue;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood fill from `from` around walls and occupied squares. Marks `seen` with
 * `stamp` and returns the lowest square reached (a canonical name for the
 * player's region: any square of the region stands for the whole region).
 */
function flood(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/**
 * Whether the box now on `t` is stuck in a frozen 2x2 block: four squares of
 * walls and boxes, at least one of them a box off its goal. Such a box can
 * never move again, so the state is dead.
 */
function isFrozen(
  w: number,
  h: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  const tx = t % w;
  const ty = Math.floor(t / w);
  const corners: Array<[number, number]> = [
    [tx - 1, ty - 1],
    [tx, ty - 1],
    [tx - 1, ty],
    [tx, ty],
  ];
  for (const [bx, by] of corners) {
    if (bx < 0 || by < 0 || bx + 1 >= w || by + 1 >= h) continue;
    const a = by * w + bx;
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a])) continue;
    if (!(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c])) continue;
    if (!(wall[d] || occupied[d])) continue;
    if (
      (occupied[a] && !goal[a]) ||
      (occupied[b] && !goal[b]) ||
      (occupied[c] && !goal[c]) ||
      (occupied[d] && !goal[d])
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Shortest walk from `from` to `to` around walls and boxes, as step letters.
 * Fixed neighbour order makes the walk deterministic.
 */
function walkTo(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/**
 * Push-optimal solution of one (sub-)level by breadth-first search over
 * pushes: states are visited in order of pushes made, so the first solved
 * state is optimal. Returns null when the state cap is hit or no solution
 * exists. `maxStates` is a work cap, never the clock.
 */
function solvePush(
  w: number,
  h: number,
  wall: Uint8Array,
  goals: number[],
  boxes0: number[],
  player0: number,
  maxStates: number,
): Solved | null {
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(w, wall, goals);
  for (const b of boxes0) if (!live[b]) return null;

  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (solved(boxes0)) return { pushes: 0, moves: '', endPlayer: player0 };

  // State key: region's lowest square, then the sorted box squares, as digits
  // in base `size` (exact for the board sizes this task draws).
  const keyOf = (boxes: number[], low: number): number => {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size); // candidate regions, kept apart from `seen`
  const stack = new Int32Array(size);
  let stamp = 0;

  const sBoxes: number[][] = [boxes0.slice()];
  const sPlayer: number[] = [player0];
  const sParent: number[] = [-1];
  const sPushFrom: number[] = [-1]; // square the pushed box started on
  const sPushDir: number[] = [-1]; // index into DIRS
  const index = new Map<number, number>();
  for (const b of boxes0) occupied[b] = 1;
  const low0 = flood(w, wall, occupied, player0, seen, ++stamp, stack);
  for (const b of boxes0) occupied[b] = 0;
  index.set(keyOf(boxes0, low0), 0);

  let found = -1;
  for (let head = 0; found < 0 && head < sBoxes.length; head++) {
    const boxes = sBoxes[head];
    for (const b of boxes) occupied[b] = 1;
    flood(w, wall, occupied, sPlayer[head], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const [dx, dy] = DIRS[k];
        const d = dy * w + dx;
        const stand = b - d;
        const dest = b + d;
        if (stand < 0 || stand >= size || dest < 0 || dest >= size) continue;
        if (seen[stand] !== here) continue; // player cannot reach the push square
        if (wall[dest] || occupied[dest] || !live[dest]) continue;
        occupied[b] = 0;
        occupied[dest] = 1;
        if (isFrozen(w, h, wall, occupied, goal, dest)) {
          occupied[dest] = 0;
          occupied[b] = 1;
          continue;
        }
        const next = boxes.slice();
        next[bi] = dest;
        next.sort((x, y) => x - y);
        const low = flood(w, wall, occupied, b, mark, ++stamp, stack);
        const key = keyOf(next, low);
        occupied[dest] = 0;
        occupied[b] = 1;
        if (index.has(key)) continue;
        if (sBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        index.set(key, sBoxes.length);
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(head);
        sPushFrom.push(b);
        sPushDir.push(k);
        if (solved(next)) {
          found = sBoxes.length - 1;
          break;
        }
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the push list, then turn each push into a walk plus one step.
  const pushes: number[] = [];
  for (let s = found; s > 0; s = sParent[s]) pushes.push(s);
  pushes.reverse();
  let moves = '';
  let player = player0;
  let boxes = boxes0.slice();
  for (const s of pushes) {
    const from = sPushFrom[s];
    const k = sPushDir[s];
    const [dx, dy, letter] = DIRS[k];
    const d = dy * w + dx;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += letter;
    player = from;
    const next = sBoxes[s];
    boxes = next.slice();
  }
  return { pushes: pushes.length, moves, endPlayer: player };
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * Split a level into rooms: cut the hallway row's dead squares (floor squares
 * from which no box can reach a goal) out of the floor; each remaining
 * connected part holding boxes is a room. Null when the level does not split
 * cleanly (a box in the hallway, a part with unequal boxes and goals, or a
 * goal left outside every part).
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(w, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (q < 0 || q >= wall.length) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) {
    if (part[b] < 0) return null;
    rooms[part[b]].boxes.push(b);
  }
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

const ROOM_STATES = 500_000;
const WHOLE_STATES = 1_000_000;

/** Solve a whole level: room by room when it splits, else one search. */
function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      // The other rooms' boxes (or, once done, the boxes resting on their
      // goals) are walls while this room is solved.
      const subWall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) subWall[b] = 1;
      }
      const solved = solvePush(
        level.w,
        level.h,
        subWall,
        room.goals,
        room.boxes,
        player,
        ROOM_STATES,
      );
      if (solved === null) {
        ok = false;
        break;
      }
      moves += solved.moves;
      player = solved.endPlayer;
      room.done = true;
    }
    if (ok) return moves;
  }
  const whole = solvePush(
    level.w,
    level.h,
    level.wall,
    level.goals,
    level.boxes,
    level.player,
    WHOLE_STATES,
  );
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(INPUT).sort()) {
  const level = parseLevel(readFileSync(join(INPUT, id, 'level.txt'), 'utf8'));
  const moves = solveLevel(level);
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'solution.txt'), `${moves}\n`);
}
