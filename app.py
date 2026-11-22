#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏎️  MCTS RACER — Monte Carlo Tree Search on an emoji racetrack 🌳
=================================================================
Left:  The car (🏎️) navigates a slalom track (🧱 walls, 🚦 start, 🏁 finish).
       · Cyan cells  = current MCTS selection path through the tree
       · White dots  = the heuristic rollout from the newest node
       · Gold cells  = the planned racing line (principal variation)
       · ⭐           = the move the car is about to take

Right: The search tree itself, drawn with emojis.
       🏎️ current state · 💭 new node · 🌱 growing · 🌳 mature · 🔥 hot
       🏁 leads to finish · 💥 leads to crash
       Gold rings/edges = planned line, cyan = live search path.

Controls:
    SPACE pause/resume    N  single MCTS iteration (even when paused)
    I     instant think   R  restart episode
    +/-   iters/frame     T  toggle tree panel
    ESC/Q quit

Emoji rendering strategy (best available wins):
    1. System colour emoji font via Pillow (recommended: pip install pillow)
    2. Twemoji PNGs from CDN (cached on disk in ./emoji_cache)
    3. Monochrome system font
Only pygame is strictly required.
"""

import heapq
import math
import os
import random
import subprocess
import sys
import traceback
import urllib.request
from collections import deque
from functools import lru_cache
from urllib.error import HTTPError

import pygame

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════
TILE           = 30
PANEL_W        = 420
HEADER_H       = 54
FOOTER_H       = 34
MARGIN         = 12
GAP            = 14

VMAX           = 3          # max velocity component
ROLLOUT_STEPS  = 40         # rollout horizon
ITERS_PER_MOVE = 320        # MCTS budget per real move
UCB_C          = 1.6        # exploration constant

TREE_BUDGET    = 110        # max nodes drawn in tree panel
TREE_MAX_KIDS  = 3          # max children drawn per node
TREE_MAX_DEPTH = 13

# Reward shaping (normalised to ≈[-1, 2] via REWARD_SCALE)
WIN_BASE, WIN_STEP_PEN     = 400.0, 4.0
CRASH_BASE, CRASH_DIST_PEN = -150.0, 2.0
PROG_W, STEP_COST, END_DIST_PEN = 3.0, 0.5, 0.3
REWARD_SCALE   = 250.0

BG   = (16, 17, 23)
TEXT = (225, 228, 235)
DIM  = (140, 145, 160)
GOLD = (255, 200, 70)
CYAN = (90, 200, 255)
RED  = (255, 110, 110)

# ═══════════════════════════════════════════════════════════════════════════
#  Track model (classic racetrack problem: position + velocity state)
# ═══════════════════════════════════════════════════════════════════════════
TRACK = [
    "############################",
    "#.....####..........####...F",
    "#.....####..........####...F",
    "#.....####..........####...F",
    "#.....####..........####...F",
    "#.....####..........####...F",
    "#S....####..........####...F",
    "#S....####...####...####...F",
    "#S...........####..........F",
    "#............####..........F",
    "#............####..........F",
    "#............####..........F",
    "#............####..........F",
    "#............####..........F",
    "############################",
]
W, H = len(TRACK[0]), len(TRACK)
TRACK_W, TRACK_H = W * TILE, H * TILE
STARTS  = [(1, 6), (1, 7), (1, 8)]
ACTIONS = [(ax, ay) for ay in (-1, 0, 1) for ax in (-1, 0, 1)]

ARROW_EMOJI = {
    (-1, -1): "↖️", (0, -1): "⬆️", (1, -1): "↗️",
    (-1, 0):  "⬅️", (0, 0):  "⏺️", (1, 0):  "➡️",
    (-1, 1):  "↙️", (0, 1):  "⬇️", (1, 1):  "↘️",
}


def _build_dist():
    """BFS distance from every road cell to the nearest finish cell."""
    dist = [[10 ** 6] * W for _ in range(H)]
    dq = deque()
    for y in range(H):
        for x in range(W):
            if TRACK[y][x] == "F":
                dist[y][x] = 0
                dq.append((x, y))
    while dq:
        x, y = dq.popleft()
        d = dist[y][x] + 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < W and 0 <= ny < H
                        and TRACK[ny][nx] in ".SF" and dist[ny][nx] > d):
                    dist[ny][nx] = d
                    dq.append((nx, ny))
    return dist


DIST = _build_dist()


def dist_at(x, y):
    if 0 <= x < W and 0 <= y < H:
        return DIST[y][x]
    return 10 ** 5


def bresenham(x0, y0, x1, y1):
    """All grid cells on the line from (x0,y0) to (x1,y1) inclusive."""
    cells = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return cells


@lru_cache(maxsize=400000)
def step_core(x, y, vx, vy, ax, ay):
    """One racetrack transition. Returns (status, x2, y2, vx2, vy2).
    Walks the full path so the car can't tunnel through thin walls."""
    vx = max(-VMAX, min(VMAX, vx + ax))
    vy = max(-VMAX, min(VMAX, vy + ay))
    nx, ny = x + vx, y + vy
    for tx, ty in bresenham(x, y, nx, ny):
        if tx == x and ty == y:
            continue
        if not (0 <= tx < W and 0 <= ty < H):
            return ("crash", tx, ty, vx, vy)
        ch = TRACK[ty][tx]
        if ch == "F":
            return ("win", tx, ty, vx, vy)
        if ch == "#":
            return ("crash", tx, ty, vx, vy)
    return ("ok", nx, ny, vx, vy)


# ═══════════════════════════════════════════════════════════════════════════
#  MCTS
# ═══════════════════════════════════════════════════════════════════════════
class Node:
    __slots__ = ("state", "parent", "action", "children", "untried",
                 "visits", "total", "terminal", "kind")

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.untried = list(ACTIONS)
        random.shuffle(self.untried)
        self.visits = 0
        self.total = 0.0
        self.terminal = False
        self.kind = "ok"

    @property
    def mean(self):
        return self.total / self.visits if self.visits else 0.0


class MCTS:
    def __init__(self, state):
        self.root = Node(state)
        self.iterations = 0
        self.node_count = 1
        self.last_path = [self.root]     # selection path of latest iteration
        self.last_rollout = []            # rollout cells of latest iteration
        self.last_reward = 0.0

    # ---- rollout policy: distance-guided with one-step crash lookahead ----
    @staticmethod
    def _rollout_action(x, y, vx, vy):
        if random.random() < 0.20:
            return random.choice(ACTIONS)
        best_a, best_k = (0, 0), 1e18
        for ax, ay in ACTIONS:
            status, x2, y2, vx2, vy2 = step_core(x, y, vx, vy, ax, ay)
            if status == "win":
                return (ax, ay)
            if status == "crash":
                k = 80.0
            else:
                k = DIST[y2][x2] + random.uniform(0.0, 1.2)
                st2 = step_core(x2, y2, vx2, vy2, 0, 0)[0]
                if st2 == "crash":
                    k += 6.0            # brake before walls
                elif st2 == "win":
                    k -= 6.0
            if k < best_k:
                best_k, best_a = k, (ax, ay)
        return best_a

    def _rollout(self, state):
        x, y, vx, vy = state
        d0 = DIST[y][x]
        cells, px, py = [], x, y
        for t in range(1, ROLLOUT_STEPS + 1):
            ax, ay = self._rollout_action(x, y, vx, vy)
            status, x, y, vx, vy = step_core(x, y, vx, vy, ax, ay)
            cells.append((x, y))
            if status == "win":
                return (WIN_BASE - WIN_STEP_PEN * t) / REWARD_SCALE, cells
            if status == "crash":
                return (CRASH_BASE - CRASH_DIST_PEN
                        * min(DIST[py][px], 60)) / REWARD_SCALE, cells
            px, py = x, y
        d_end = DIST[py][px]
        raw = PROG_W * (d0 - d_end) - STEP_COST * t - END_DIST_PEN * d_end
        return raw / REWARD_SCALE, cells

    @staticmethod
    def _terminal_value(kind, state):
        if kind == "win":
            return WIN_BASE / REWARD_SCALE
        return (CRASH_BASE - CRASH_DIST_PEN
                * min(dist_at(state[0], state[1]), 60)) / REWARD_SCALE

    def iterate(self):
        """One full MCTS iteration: select → expand → simulate → backprop."""
        node = self.root
        path = [node]
        # SELECT (UCB1)
        while not node.terminal and not node.untried and node.children:
            node = max(node.children, key=lambda c:
                       c.total / c.visits
                       + UCB_C * math.sqrt(math.log(node.visits + 1) / c.visits))
            path.append(node)
        # EXPAND
        if node.terminal:
            reward = self._terminal_value(node.kind, node.state)
            cells = []
        else:
            action = node.untried.pop()
            x, y, vx, vy = node.state
            status, x2, y2, vx2, vy2 = step_core(
                x, y, vx, vy, action[0], action[1])
            child = Node((x2, y2, vx2, vy2), node, action)
            child.terminal = status != "ok"
            child.kind = status
            node.children.append(child)
            self.node_count += 1
            path.append(child)
            if child.terminal:
                reward = self._terminal_value(status, child.state)
                cells = []
            else:
                reward, cells = self._rollout(child.state)
        # BACKPROPAGATE
        for n in path:
            n.visits += 1
            n.total += reward
        self.iterations += 1
        self.last_path = path
        self.last_rollout = cells
        self.last_reward = reward

    def best_child(self):
        if not self.root.children:
            return None
        return max(self.root.children, key=lambda c: c.visits)

    def principal_variation(self, limit=14):
        seq, node = [], self.root
        while node.children and len(seq) < limit:
            node = max(node.children, key=lambda c: c.visits)
            seq.append(node)
        return seq

    def advance(self, action):
        """Re-root the tree after a real move (keeps the useful subtree)."""
        for c in self.root.children:
            if c.action == action:
                self.root = c
                self.root.parent = None
                break
        else:
            x, y, vx, vy = self.root.state
            _, x2, y2, vx2, vy2 = step_core(x, y, vx, vy, action[0], action[1])
            self.root = Node((x2, y2, vx2, vy2))
        self.iterations = 0
        self.last_path = [self.root]
        self.last_rollout = []
        self.node_count = self._count(self.root)

    @staticmethod
    def _count(root):
        n, stack = 0, [root]
        while stack:
            node = stack.pop()
            n += 1
            stack.extend(node.children)
        return n


# ═══════════════════════════════════════════════════════════════════════════
#  Emoji rendering (PIL colour fonts → Twemoji CDN → monochrome → fallback)
# ═══════════════════════════════════════════════════════════════════════════
class EmojiRenderer:
    TWEMOJI = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/{}.png"
    CACHE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "emoji_cache")

    FONT_CANDIDATES = [
        r"C:\Windows\Fonts\seguiemj.ttf",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts\seguiemj.ttf"),
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/noto-emoji/NotoColorEmoji.ttf",
        "/usr/share/fonts/truetype/noto-emoji/NotoColorEmoji.ttf",
        "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
        "/usr/share/fonts/google-noto/NotoColorEmoji.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/System/Library/Fonts/Apple Color Emoji.ttf",
    ]

    def __init__(self):
        self.cache = {}
        self.pil_font = None
        self.pil_size = 0
        self.pil_color = False
        self.pg_font = None
        self.backend = "pygame font (basic fallback)"
        self.used_twemoji = False
        self.net_failures = 0
        self.net_disabled = False
        self._tw_mem = {}
        self._setup_pil()

    def _find_font(self):
        env = os.environ.get("EMOJI_FONT")
        if env and os.path.isfile(env):
            return env
        for p in self.FONT_CANDIDATES:
            if os.path.isfile(p):
                return p
        try:
            out = subprocess.run(
                ["fc-list"], capture_output=True, text=True, timeout=5).stdout or ""
            for line in out.splitlines():
                if "emoji" in line.lower():
                    path = line.split(":", 1)[0].strip()
                    if path and os.path.isfile(path):
                        return path
        except Exception:
            pass
        return None

    def _setup_pil(self):
        if not PIL_OK:
            return
        path = self._find_font()
        if not path:
            return
        for size in (109, 137, 128, 96, 72, 64):
            try:
                font = ImageFont.truetype(path, size)
            except Exception:
                continue
            try:
                img = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
                ImageDraw.Draw(img).text(
                    (size // 2, size // 2), "🏁",
                    font=font, embedded_color=True)
            except Exception:
                continue
            if img.getbbox() is None:
                continue
            self.pil_font = font
            self.pil_size = size
            self.pil_color = self._has_color(img)
            kind = "colour" if self.pil_color else "mono"
            self.backend = "system %s emoji font (%s)" % (kind, path)
            return

    @staticmethod
    def _has_color(img):
        try:
            for r, g, b, a in img.convert("RGBA").getdata():
                if a > 60 and max(r, g, b) - min(r, g, b) > 40:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _code_variants(emoji):
        full = "-".join("%x" % ord(c) for c in emoji)
        stripped = "-".join("%x" % ord(c) for c in emoji if ord(c) != 0xFE0F)
        return [full] if full == stripped else [full, stripped]

    def _load_twemoji(self, emoji):
        for code in self._code_variants(emoji):
            path = os.path.join(self.CACHE_DIR, code + ".png")
            if os.path.isfile(path):
                try:
                    return Image.open(path).convert("RGBA")
                except Exception:
                    pass
            if self.net_disabled:
                continue
            try:
                with urllib.request.urlopen(
                        self.TWEMOJI.format(code), timeout=5) as r:
                    data = r.read()
            except HTTPError:
                continue
            except Exception:
                self.net_failures += 1
                if self.net_failures >= 2:
                    self.net_disabled = True
                continue
            try:
                os.makedirs(self.CACHE_DIR, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(data)
                return Image.open(path).convert("RGBA")
            except Exception:
                continue
        return None

    def _tw_image(self, emoji):
        if emoji not in self._tw_mem:
            img = self._load_twemoji(emoji)
            if img is not None:
                self.used_twemoji = True
            self._tw_mem[emoji] = img
        return self._tw_mem[emoji]

    def _pil_render(self, emoji):
        img = Image.new("RGBA", (self.pil_size * 2, self.pil_size * 2),
                        (0, 0, 0, 0))
        try:
            ImageDraw.Draw(img).text(
                (self.pil_size // 2, self.pil_size // 2), emoji,
                font=self.pil_font, embedded_color=True)
        except Exception:
            try:
                ImageDraw.Draw(img).text(
                    (self.pil_size // 2, self.pil_size // 2), emoji,
                    font=self.pil_font)
            except Exception:
                return None
        bbox = img.getbbox()
        return img.crop(bbox) if bbox else None

    @staticmethod
    def _to_surface(img, size):
        resample = getattr(Image, "LANCZOS", None)
        if resample is None:
            resample = Image.Resampling.LANCZOS
        img = img.copy()
        img.thumbnail((size, size), resample)
        w, h = img.size
        square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        square.paste(img, ((size - w) // 2, (size - h) // 2), img)
        surf = pygame.image.fromstring(square.tobytes(), square.size, "RGBA")
        try:
            if pygame.display.get_surface() is not None:
                surf = surf.convert_alpha()
        except Exception:
            pass
        return surf

    def get(self, emoji, size):
        size = max(4, int(size))
        key = (emoji, size)
        s = self.cache.get(key)
        if s is None:
            s = self._render(emoji, size)
            self.cache[key] = s
        return s

    def _render(self, emoji, size):
        if self.pil_font is not None and self.pil_color:
            img = self._pil_render(emoji)
            if img is not None:
                return self._to_surface(img, size)
        img = self._tw_image(emoji)
        if img is not None:
            return self._to_surface(img, size)
        if self.pil_font is not None:
            img = self._pil_render(emoji)
            if img is not None:
                return self._to_surface(img, size)
        if self.pg_font is None:
            self.pg_font = pygame.font.SysFont(
                "notocoloremoji,seguiemj,segoeuiemoji,"
                "applesymbols,dejavusans,arial", 32)
        try:
            s = self.pg_font.render(emoji, True, (235, 235, 235))
        except Exception:
            s = pygame.Surface((size, size), pygame.SRCALPHA)
        if s.get_size() != (size, size):
            s = pygame.transform.smoothscale(s, (size, size))
        return s

    def describe(self):
        d = self.backend
        if self.used_twemoji:
            d += " + twemoji CDN/cache"
        return d


# ═══════════════════════════════════════════════════════════════════════════
#  Application
# ═══════════════════════════════════════════════════════════════════════════
PRELOAD = ["🧱", "🏁", "🏎️", "🚦", "💥", "🎉", "✨", "🎊", "💭", "🌱", "🌳",
           "🔥", "⭐", "🧠", "💨", "🏆", "🥇", "💀",
           "⬆️", "⬇️", "⬅️", "➡️", "↖️", "↗️", "↘️", "↙️", "⏺️"]

SCREEN_W = MARGIN + TRACK_W + GAP + PANEL_W + MARGIN
SCREEN_H = MARGIN + HEADER_H + 8 + TRACK_H + 10 + FOOTER_H + MARGIN


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("🏎️ MCTS Racer 🌳 — emoji edition")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont(
            "arialblack,arial,helvetica,dejavusans", 24, bold=True)
        self.font_md = pygame.font.SysFont(
            "arial,helvetica,dejavusans", 16, bold=True)
        self.font_sm = pygame.font.SysFont(
            "consolas,menlo,dejavusansmono,monospace", 14)

        self.emo = EmojiRenderer()
        for e in PRELOAD:
            self.emo.get(e, 24)
        print("[emoji] backend: %s" % self.emo.describe())

        self.track_rect = pygame.Rect(
            MARGIN, MARGIN + HEADER_H + 8, TRACK_W, TRACK_H)
        self.panel_rect = pygame.Rect(
            self.track_rect.right + GAP, MARGIN, PANEL_W, SCREEN_H - 2 * MARGIN)
        self.footer_rect = pygame.Rect(
            MARGIN, self.track_rect.bottom + 10,
            TRACK_W + GAP + PANEL_W, FOOTER_H)

        self.track_surf = self._build_track()
        self.car_size = TILE + 10
        self.car_base = pygame.transform.flip(
            self.emo.get("🏎️", self.car_size), True, False)
        self.car_rot = {}

        self.wins = self.crashes = self.attempts = 0
        self.best_steps = None
        self.total_iters = 0
        self.particles = []
        self.trail = []
        self.show_tree = True
        self.paused = False
        self.iters_per_frame = 8
        self.frame = 0
        self.tree_hover = None
        self.running = True
        self._reset_episode()

    # ------------------------------------------------------------- track art
    def _build_track(self):
        surf = pygame.Surface((TRACK_W, TRACK_H), pygame.SRCALPHA)
        for y in range(H):
            for x in range(W):
                r = pygame.Rect(x * TILE, y * TILE, TILE, TILE)
                ch = TRACK[y][x]
                if ch == "#":
                    surf.fill((40, 26, 22), r)
                    surf.blit(self.emo.get("🧱", TILE), r.topleft)
                elif ch == "F":
                    surf.fill((74, 58, 16), r)
                    self._blit_emoji(surf, "🏁", int(TILE * 0.92),
                                     r.centerx, r.centery)
                elif ch == "S":
                    surf.fill((26, 58, 44), r)
                    self._blit_emoji(surf, "🚦", int(TILE * 0.7),
                                     r.centerx, r.centery)
                else:
                    base = (33, 36, 48) if (x + y) % 2 else (29, 32, 43)
                    surf.fill(base, r)
                pygame.draw.rect(surf, (44, 48, 62), r, 1)
        pygame.draw.rect(surf, (90, 95, 115), surf.get_rect(), 2)
        return surf

    # -------------------------------------------------------------- helpers
    def _cell_center(self, x, y):
        return (self.track_rect.x + x * TILE + TILE // 2,
                self.track_rect.y + y * TILE + TILE // 2)

    @staticmethod
    def _local_center(x, y):
        return (x * TILE + TILE // 2, y * TILE + TILE // 2)

    def _blit_emoji(self, dest, emoji, size, cx, cy):
        s = self.emo.get(emoji, size)
        dest.blit(s, s.get_rect(center=(int(cx), int(cy))))

    def _car_surface(self, angle):
        a = int(round(angle / 15.0) * 15) % 360
        s = self.car_rot.get(a)
        if s is None:
            s = pygame.transform.rotate(self.car_base, a)
            self.car_rot[a] = s
        return s

    @staticmethod
    def _lerp_angle(a0, a1, f):
        d = (a1 - a0 + 180) % 360 - 180
        return a0 + d * f

    @staticmethod
    def _path_point(pts, f):
        if len(pts) < 2:
            return pts[0]
        lengths = [math.hypot(pts[i + 1][0] - pts[i][0],
                              pts[i + 1][1] - pts[i][1])
                   for i in range(len(pts) - 1)]
        total = sum(lengths) or 1
        target = f * total
        acc = 0.0
        for i, seg in enumerate(lengths):
            if acc + seg >= target or i == len(lengths) - 1:
                t = 0.0 if seg == 0 else (target - acc) / seg
                t = max(0.0, min(1.0, t))
                return (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t,
                        pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t)
            acc += seg
        return pts[-1]

    def _rich(self, dest, x, y, tokens, font, right=None, color=TEXT):
        """Draw a mixed emoji/text line. Returns end x position."""
        widths = []
        for tok in tokens:
            if tok[0] == "t":
                widths.append(font.size(tok[1])[0])
            else:
                widths.append((tok[2] if len(tok) > 2
                               else font.get_height()) + 1)
        total = sum(widths)
        if right is not None:
            x = right - total
        for tok, w in zip(tokens, widths):
            if tok[0] == "t":
                col = tok[2] if len(tok) > 2 else color
                dest.blit(font.render(tok[1], True, col), (x, y))
            else:
                size = tok[2] if len(tok) > 2 else font.get_height()
                s = self.emo.get(tok[1], size)
                dest.blit(s, s.get_rect(
                    midleft=(x, y + font.get_height() // 2)))
            x += w
        return x

    # ------------------------------------------------------------ episode
    def _reset_episode(self):
        x, y = random.choice(STARTS)
        state = (x, y, 0, 0)
        self.mcts = MCTS(state)
        self.car_state = state
        self.car_px = self._cell_center(x, y)
        self.car_angle = 0.0
        self.episode_steps = 0
        self.mode = "think"
        self.move = None
        self.trail = []
        self.attempts += 1

    def _start_move(self, child):
        x, y, vx, vy = self.car_state
        ax, ay = child.action
        status, x2, y2, vx2, vy2 = step_core(x, y, vx, vy, ax, ay)
        cells = bresenham(x, y, x2, y2)[1:]
        pts = [self._cell_center(x, y)] + \
              [self._cell_center(cx, cy) for cx, cy in cells]
        if len(pts) == 1:
            pts.append(pts[0])
        angle1 = (self.car_angle if (vx2 == 0 and vy2 == 0)
                  else -math.degrees(math.atan2(vy2, vx2)))
        self.move = {
            "action": child.action, "status": status,
            "new_state": (x2, y2, vx2, vy2),
            "pts": pts, "t": 0.0,
            "dur": max(8, 5 * len(cells)),
            "angle0": self.car_angle, "angle1": angle1,
        }
        self.mode = "move"

    def _update_move(self):
        m = self.move
        m["t"] += 1.0
        f = min(1.0, m["t"] / m["dur"])
        self.car_px = self._path_point(m["pts"], f)
        self.car_angle = self._lerp_angle(m["angle0"], m["angle1"], f)
        self.trail.append((self.car_px, self.frame))
        vx, vy = m["new_state"][2], m["new_state"][3]
        if max(abs(vx), abs(vy)) >= 2 and self.frame % 2 == 0:
            self.particles.append({
                "emoji": "💨",
                "x": self.car_px[0] + random.uniform(-4, 4),
                "y": self.car_px[1] + random.uniform(-4, 4),
                "vx": -vx * 0.5, "vy": -vy * 0.5,
                "life": 14, "life0": 14, "size": 16})
        if f >= 1.0:
            self._finish_move()

    def _finish_move(self):
        m = self.move
        self.episode_steps += 1
        self.car_state = m["new_state"]
        if m["status"] == "win":
            self.wins += 1
            if self.best_steps is None or self.episode_steps < self.best_steps:
                self.best_steps = self.episode_steps
            self.mode = "win"
            self.mode_until = self.frame + 100
            self._confetti(m["pts"][-1])
        elif m["status"] == "crash":
            self.crashes += 1
            self.mode = "crash"
            self.mode_until = self.frame + 60
            self._boom(m["pts"][-1])
        else:
            self.mcts.advance(m["action"])
            self.mode = "think"

    # ---------------------------------------------------------- particles
    def _confetti(self, pos):
        for _ in range(26):
            self.particles.append({
                "emoji": random.choice(["🎉", "✨", "🎊", "⭐"]),
                "x": pos[0] + random.uniform(-24, 24),
                "y": pos[1] + random.uniform(-24, 24),
                "vx": random.uniform(-3, 3),
                "vy": random.uniform(-6.5, -1.5),
                "life": random.randint(55, 95), "life0": 95,
                "size": random.choice([18, 22, 26])})

    def _boom(self, pos):
        for _ in range(12):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(0.5, 3.0)
            self.particles.append({
                "emoji": "💥",
                "x": pos[0] + random.uniform(-6, 6),
                "y": pos[1] + random.uniform(-6, 6),
                "vx": math.cos(ang) * spd,
                "vy": math.sin(ang) * spd - 1,
                "life": random.randint(24, 40), "life0": 40,
                "size": random.choice([20, 26, 32])})

    def _update_particles(self):
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.15
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    # ------------------------------------------------------------- events
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_r:
                    self._reset_episode()
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.iters_per_frame = min(60, self.iters_per_frame + 2)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.iters_per_frame = max(1, self.iters_per_frame - 2)
                elif e.key == pygame.K_t:
                    self.show_tree = not self.show_tree
                elif e.key == pygame.K_i:
                    while (self.mode == "think"
                           and self.mcts.iterations < ITERS_PER_MOVE):
                        self.mcts.iterate()
                        self.total_iters += 1
                elif e.key == pygame.K_n:
                    if self.mode == "think":
                        self.mcts.iterate()
                        self.total_iters += 1

    # ------------------------------------------------------------- update
    def update(self):
        if self.paused:
            return
        self.frame += 1
        if self.mode == "think":
            for _ in range(self.iters_per_frame):
                if self.mcts.iterations < ITERS_PER_MOVE:
                    self.mcts.iterate()
                    self.total_iters += 1
            if self.mcts.iterations >= ITERS_PER_MOVE:
                child = self.mcts.best_child()
                if child is None:
                    self.mcts.iterate()
                    self.total_iters += 1
                else:
                    self._start_move(child)
        elif self.mode == "move":
            self._update_move()
        elif self.frame >= self.mode_until:
            self._reset_episode()
        self._update_particles()
        self.trail = [t for t in self.trail if self.frame - t[1] < 26]

    # ------------------------------------------------------------ drawing
    def draw(self):
        self.screen.fill(BG)
        self._draw_header()
        self.screen.blit(self.track_surf, self.track_rect.topleft)
        self._draw_track_overlays()
        self._draw_trail()
        self._draw_car()
        self._draw_particles()
        if self.mode == "win":
            self._draw_win_banner()
        if self.show_tree:
            self._draw_panel()
        else:
            hint = self.font_sm.render("tree hidden — press T", True, DIM)
            self.screen.blit(hint, hint.get_rect(center=self.panel_rect.center))
        self._draw_footer()
        if self.paused:
            self._draw_pause_overlay()
        pygame.display.flip()

    def _draw_header(self):
        scr = self.screen
        self._blit_emoji(scr, "🏎️", 30, MARGIN + 16, MARGIN + HEADER_H // 2)
        scr.blit(self.font_lg.render("MCTS RACER", True, TEXT),
                 (MARGIN + 40, MARGIN + 2))
        scr.blit(self.font_md.render(
            "Monte Carlo Tree Search — emoji edition", True, DIM),
            (MARGIN + 42, MARGIN + 31))
        best = str(self.best_steps) if self.best_steps is not None else "—"
        self._rich(scr, 0, MARGIN + 18, [
            ("e", "🥇", 16), ("t", " wins %d" % self.wins, TEXT),
            ("t", "    ", DIM),
            ("e", "💀", 16), ("t", " crashes %d" % self.crashes, TEXT),
            ("t", "    ", DIM),
            ("e", "🧠", 16), ("t", " iters %d" % self.total_iters, TEXT),
            ("t", "    ", DIM),
            ("e", "🏆", 16), ("t", " best %s steps" % best, TEXT),
        ], self.font_sm, right=SCREEN_W - MARGIN)

    def _draw_track_overlays(self):
        ox, oy = self.track_rect.topleft
        ov = pygame.Surface((TRACK_W, TRACK_H), pygame.SRCALPHA)
        m = self.mcts
        pv = []
        if self.mode == "think":
            # rollout dots
            for cx, cy in m.last_rollout:
                if 0 <= cx < W and 0 <= cy < H:
                    pygame.draw.circle(ov, (255, 255, 255, 60),
                                       self._local_center(cx, cy), 2)
            # selection path cells
            for n in m.last_path[1:]:
                cx, cy = n.state[0], n.state[1]
                if 0 <= cx < W and 0 <= cy < H:
                    r = pygame.Rect(cx * TILE + 2, cy * TILE + 2,
                                    TILE - 4, TILE - 4)
                    pygame.draw.rect(ov, (90, 200, 255, 60), r, 2)
            # planned racing line
            pv = m.principal_variation(12)
            if pv:
                pts = [self._local_center(n.state[0], n.state[1]) for n in pv]
                for pt in pts:
                    pygame.draw.circle(ov, (255, 200, 70, 200), pt, 4)
                if len(pts) > 1:
                    pygame.draw.lines(ov, (255, 200, 70, 110), False, pts, 2)
        # hover link from tree panel
        if self.tree_hover is not None:
            cx, cy = self.tree_hover.state[0], self.tree_hover.state[1]
            if 0 <= cx < W and 0 <= cy < H:
                r = pygame.Rect(cx * TILE + 1, cy * TILE + 1, TILE - 2, TILE - 2)
                pygame.draw.rect(ov, (120, 255, 220, 220), r, 2)
        self.screen.blit(ov, (ox, oy))
        # star on the chosen next cell
        if pv:
            px, py = pv[0].state[0], pv[0].state[1]
            if 0 <= px < W and 0 <= py < H:
                lc = self._local_center(px, py)
                self._blit_emoji(self.screen, "⭐", 20, ox + lc[0], oy + lc[1])

    def _draw_trail(self):
        if len(self.trail) > 1:
            pts = [(int(p[0]), int(p[1])) for p, _ in self.trail]
            pygame.draw.lines(self.screen, (185, 150, 60), False, pts, 2)

    def _draw_car(self):
        if self.mode in ("crash", "win"):
            return
        if self.mode == "think":
            vx, vy = self.car_state[2], self.car_state[3]
            if vx or vy:
                self.car_angle = -math.degrees(math.atan2(vy, vx))
        s = self._car_surface(self.car_angle)
        r = s.get_rect(center=(int(self.car_px[0]), int(self.car_px[1])))
        sh = pygame.Surface((r.w, max(4, r.h // 2)), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, 70), sh.get_rect())
        self.screen.blit(sh, (r.x, r.bottom - sh.get_height() // 2))
        self.screen.blit(s, r)

    def _draw_particles(self):
        for p in self.particles:
            s = self.emo.get(p["emoji"], p["size"])
            frac = p["life"] / p["life0"]
            if frac < 0.45:
                s = s.copy()
                s.fill((255, 255, 255, int(255 * frac / 0.45)),
                       special_flags=pygame.BLEND_RGBA_MULT)
            self.screen.blit(s, s.get_rect(
                center=(int(p["x"]), int(p["y"]))))

    def _draw_pause_overlay(self):
        s = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        s.fill((0, 0, 0, 110))
        self.screen.blit(s, (0, 0))
        t = self.font_lg.render("PAUSED", True, GOLD)
        self.screen.blit(t, t.get_rect(
            center=(SCREEN_W // 2, SCREEN_H // 2)))

    def _draw_win_banner(self):
        tr = self.track_rect
        s = pygame.Surface((tr.w, tr.h), pygame.SRCALPHA)
        s.fill((0, 0, 0, 120))
        self.screen.blit(s, tr.topleft)
        self._blit_emoji(self.screen, "🏆", 64, tr.centerx, tr.centery - 18)
        t = self.font_lg.render(
            "LAP COMPLETE — %d steps" % self.episode_steps, True, GOLD)
        self.screen.blit(t, t.get_rect(
            center=(tr.centerx, tr.centery + 30)))

    def _draw_footer(self):
        scr, r, m = self.screen, self.footer_rect, self.mcts
        pygame.draw.rect(scr, (22, 23, 30), r, border_radius=8)
        pygame.draw.rect(scr, (58, 62, 80), r, 1, border_radius=8)
        if self.mode == "think":
            toks = [("e", "🧠", 15),
                    ("t", " thinking… %d/%d iters (%d/frame) · "
                          "last rollout %+.2f · tree %d nodes"
                     % (m.iterations, ITERS_PER_MOVE, self.iters_per_frame,
                        m.last_reward, m.node_count), TEXT)]
        elif self.mode == "move":
            v = self.move["new_state"]
            toks = [("e", "🏎️", 15),
                    ("t", " executing action %s → new velocity (%d,%d)"
                     % (self.move["action"], v[2], v[3]), TEXT)]
        elif self.mode == "crash":
            toks = [("e", "💥", 15),
                    ("t", " CRASHED into the wall — respawning…", RED)]
        else:
            toks = [("e", "🎉", 15),
                    ("t", " FINISH! lap in %d steps (best %s)"
                     % (self.episode_steps, self.best_steps), GOLD)]
        if self.paused:
            toks.append(("t", "   [PAUSED]", GOLD))
        self._rich(scr, r.x + 12, r.y + 9, toks, self.font_sm)
        ctrl = ("SPACE pause · N step · I insta-think · "
                "R restart · +/- speed · T tree")
        s = self.font_sm.render(ctrl, True, DIM)
        scr.blit(s, (r.right - 12 - s.get_width(), r.y + 10))

    # -------------------------------------------------------- tree panel
    def _draw_panel(self):
        p, scr = self.panel_rect, self.screen
        pygame.draw.rect(scr, (22, 23, 30), p, border_radius=12)
        pygame.draw.rect(scr, (58, 62, 80), p, 1, border_radius=12)
        x0 = p.x + 14
        self._rich(scr, x0, p.y + 8,
                   [("e", "🌳", 20), ("t", "  SEARCH TREE", GOLD)], self.font_md)
        s = self.font_sm.render(
            "%d/%d iters" % (self.mcts.iterations, ITERS_PER_MOVE), True, DIM)
        scr.blit(s, (p.right - 14 - s.get_width(), p.y + 12))

        # principal variation as emoji arrows
        pv = self.mcts.principal_variation(14)
        toks = [("t", "plan: ", DIM)]
        for n in pv:
            toks.append(("e", ARROW_EMOJI.get(n.action, "⏺️"), 15))
        if not pv:
            toks.append(("t", "—", DIM))
        self._rich(scr, x0, p.y + 32, toks, self.font_sm)

        ty, th = p.y + 54, p.h - 54 - 46 - 158
        self._draw_tree(x0, ty, p.w - 24, th)

        ly = ty + th + 6
        self._rich(scr, x0, ly, [
            ("e", "🏎️", 13), ("t", " now   ", DIM),
            ("e", "💭", 13), ("t", " new   ", DIM),
            ("e", "🌱", 13), ("t", " growing   ", DIM),
            ("e", "🌳", 13), ("t", " mature   ", DIM),
            ("e", "🔥", 13), ("t", " hot", DIM)], self.font_sm)
        self._rich(scr, x0, ly + 17, [
            ("e", "🏁", 13), ("t", " finish   ", DIM),
            ("e", "💥", 13), ("t", " crash   ", DIM),
            ("e", "⭐", 13), ("t", " planned · ", DIM),
            ("t", "cyan = live search path", CYAN)], self.font_sm)

        self._draw_actions_table(x0, ly + 38)

    def _draw_tree(self, x0, y0, w, h):
        scr, m = self.screen, self.mcts
        root = m.root
        self.tree_hover = None

        # ---- pick nodes to show: best-first by visit count ----
        shown = {id(root): (root, 0)}
        order = [root]
        heap = [(-root.visits, 0, root, 0)]
        seq = 0
        while heap and len(order) < TREE_BUDGET:
            _, _, node, depth = heapq.heappop(heap)
            if depth >= TREE_MAX_DEPTH:
                continue
            kids = sorted(node.children, key=lambda c: c.visits,
                          reverse=True)[:TREE_MAX_KIDS]
            for c in kids:
                if len(order) >= TREE_BUDGET:
                    break
                if id(c) in shown:
                    continue
                shown[id(c)] = (c, depth + 1)
                order.append(c)
                seq += 1
                heapq.heappush(heap, (-c.visits, seq, c, depth + 1))

        kids_of = {id(n): [c for c in n.children if id(c) in shown]
                   for n in order}

        # ---- tidy vertical layout (leaves get slots, parents get means) ----
        yslot, cursor, maxd = {}, [0.0], [0]

        def place(n):
            kids = kids_of[id(n)]
            if not kids:
                yslot[id(n)] = float(cursor[0])
                cursor[0] += 1.0
            else:
                first = last = None
                for c in kids:
                    yy = place(c)
                    first = yy if first is None else first
                    last = yy
                yslot[id(n)] = (first + last) / 2.0
            maxd[0] = max(maxd[0], shown[id(n)][1])
            return yslot[id(n)]

        place(root)

        leaves = max(1, int(cursor[0]))
        row = min(26.0, h / leaves)
        colw = min(42.0, (w - 24) / max(1, maxd[0]))
        em = max(8, min(24, int(row * 0.95)))

        pos = {}
        for n in order:
            d = shown[id(n)][1]
            pos[id(n)] = (int(x0 + 12 + d * colw + em // 2),
                          int(y0 + (yslot[id(n)] + 0.5) * row) + 2)

        # principal variation + selection path (within shown nodes)
        pv_ids = set()
        sel_ids = set(id(n) for n in m.last_path)
        n = root
        while kids_of.get(id(n)):
            n = max(kids_of[id(n)], key=lambda c: c.visits)
            pv_ids.add(id(n))

        # ---- edges ----
        for n in order:
            for c in kids_of[id(n)]:
                x1, y1 = pos[id(n)]
                x2, y2 = pos[id(c)]
                onpv = id(n) in pv_ids and id(c) in pv_ids
                onsel = id(c) in sel_ids
                col = GOLD if onpv else (CYAN if onsel else (70, 75, 95))
                pygame.draw.line(scr, col,
                                 (x1 + em // 2, y1),
                                 (x2 - em // 2 - 1, y2),
                                 2 if onpv else 1)

        # ---- nodes ----
        mouse = pygame.mouse.get_pos()
        hover, hover_d = None, 15
        for n in order:
            x, y = pos[id(n)]
            if n is root:
                e, sz = "🏎️", em + 6
            elif n.terminal:
                e, sz = ("🏁" if n.kind == "win" else "💥"), em
            elif n.visits <= 1:
                e, sz = "💭", em
            elif n.visits < 8:
                e, sz = "🌱", em
            elif n.visits < 40:
                e, sz = "🌳", em
            else:
                e, sz = "🔥", em
            s = self.emo.get(e, sz)
            scr.blit(s, s.get_rect(center=(x, y)))
            if id(n) in pv_ids:
                pygame.draw.circle(scr, GOLD, (x, y), int(sz * 0.75), 1)
            if id(n) in sel_ids:
                pygame.draw.circle(scr, CYAN, (x, y), int(sz * 0.75) + 1, 1)
            d = math.hypot(mouse[0] - x, mouse[1] - y)
            if d < hover_d:
                hover_d, hover = d, n

        # pulse on the newest expanded node
        newest = m.last_path[-1] if m.last_path else None
        if newest is not None and id(newest) in pos:
            nx, ny = pos[id(newest)]
            pr = int(em + 4 + 3 * math.sin(self.frame * 0.3))
            pygame.draw.circle(scr, (255, 255, 255), (nx, ny), pr, 1)

        s = self.font_sm.render(
            "showing %d/%d nodes" % (len(order), m.node_count), True, DIM)
        scr.blit(s, (x0 + w - s.get_width(), y0))
        self.tree_hover = hover
        if hover is not None:
            self._draw_tree_tooltip(hover, pos[id(hover)])

    def _draw_tree_tooltip(self, node, xy):
        scr = self.screen
        lines = []
        if node.action:
            lines.append([("e", ARROW_EMOJI.get(node.action, "⏺️"), 13),
                          ("t", "  action %s" % (node.action,), TEXT)])
        lines.append([("t", "cell (%d,%d)  vel (%d,%d)"
                       % (node.state[0], node.state[1],
                          node.state[2], node.state[3]), TEXT)])
        lines.append([("t", "visits %d   mean %+.2f"
                       % (node.visits, node.mean), TEXT)])
        wbox = 0
        for ln in lines:
            lw = sum(self.font_sm.size(t[1])[0] if t[0] == "t" else 16
                     for t in ln)
            wbox = max(wbox, lw)
        wbox += 16
        hbox = len(lines) * 17 + 10
        tx = min(max(xy[0] + 12, self.panel_rect.x + 6),
                 self.panel_rect.right - wbox - 6)
        ty = xy[1] - hbox - 6
        if ty < self.panel_rect.y + 40:
            ty = xy[1] + 14
        box = pygame.Rect(tx, ty, wbox, hbox)
        bg = pygame.Surface((wbox, hbox), pygame.SRCALPHA)
        bg.fill((10, 10, 16, 235))
        scr.blit(bg, box.topleft)
        pygame.draw.rect(scr, (120, 200, 160), box, 1, border_radius=4)
        for i, ln in enumerate(lines):
            self._rich(scr, box.x + 8, box.y + 5 + i * 17, ln, self.font_sm)

    def _draw_actions_table(self, x0, y0):
        scr = self.screen
        self._rich(scr, x0, y0,
                   [("e", "🎯", 18),
                    ("t", "  ROOT ACTIONS  (bar = visits)", GOLD)],
                   self.font_md)
        kids = sorted(self.mcts.root.children,
                      key=lambda c: c.visits, reverse=True)
        if not kids:
            scr.blit(self.font_sm.render("no data yet…", True, DIM),
                     (x0 + 10, y0 + 26))
            return
        maxv = max(c.visits for c in kids) or 1
        for i, c in enumerate(kids[:6]):
            y = y0 + 26 + i * 19
            if c.terminal:
                icon = "🏁" if c.kind == "win" else "💥"
            else:
                icon = ARROW_EMOJI.get(c.action, "⏺️")
            self._blit_emoji(scr, icon, 15, x0 + 10, y + 9)
            bw = max(2, int(110 * c.visits / maxv))
            pygame.draw.rect(scr, GOLD if i == 0 else (95, 100, 125),
                             (x0 + 26, y + 3, bw, 12), border_radius=3)
            txt = "N=%-4d μ=%+.2f" % (c.visits, c.mean)
            scr.blit(self.font_sm.render(txt, True,
                                         GOLD if i == 0 else TEXT),
                     (x0 + 148, y + 2))
            if i == 0:
                self._blit_emoji(scr, "⭐", 14, x0 + 306, y + 9)
        if len(kids) > 6:
            scr.blit(self.font_sm.render(
                "…and %d more" % (len(kids) - 6), True, DIM),
                (x0 + 26, y0 + 26 + 6 * 19))

    # ---------------------------------------------------------------- main
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


def main():
    try:
        App().run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()