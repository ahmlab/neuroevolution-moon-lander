"""
Evaluate the evolved lander. evaluate.py was vibe coded for record and visual purposes.

Examples:
    python evaluate.py watch  --seeds 1000 1001 1002               # just watch them land
    python evaluate.py record --n 6 --grid --cols 3                 # 6 landings, one per cell
    python evaluate.py record --n 27 --grid --cols 3 --rows 3       # 3x3 grid, 3 landings per cell
    python evaluate.py record --seeds 1003 1042 1077                # landings one after another
    python evaluate.py record --n 27 --grid --cols 3 --rows 3 --out landings.mp4   # video (needs imageio-ffmpeg)
    python evaluate.py test   --n 100                               # stats on 100 unseen seeds
"""
import argparse
import math
import time
import random

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from landing_tools import forward

ENV_FPS = 50  # LunarLander's native frame rate


def make_env(render_mode=None):
    return gym.make("LunarLander-v3", continuous=True, render_mode=render_mode)


def play(env, weights, seed, record=False, frame_skip=1, max_steps=1000):
    """Run one episode. Returns (total_reward, frames)."""
    obs, _ = env.reset(seed=seed)
    total, frames = 0.0, []
    for step in range(max_steps):
        action = forward(obs, weights)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if record and step % frame_skip == 0:
            frames.append(env.render())
        if terminated or truncated:
            break
    if record:
        frames.append(env.render())  # make sure the final frame is included
    return total, frames


def get_font(size):
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def prepare(frame, text, scale, font):
    """Resize a frame and write a label on it."""
    img = Image.fromarray(frame)
    if scale != 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    ImageDraw.Draw(img).text((8, 6), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def build_streams(episodes, n_cells, hold):
    """
    Distribute episodes across cells. Each cell plays its episodes back to back.
    Every new episode goes to the cell whose stream is currently shortest,
    so all cells finish at roughly the same time.
    """
    streams = [[] for _ in range(n_cells)]
    for ep in episodes:
        cell = min(range(n_cells), key=lambda i: len(streams[i]))
        streams[cell] += ep + [ep[-1]] * hold   # pause on the final frame before the next one
    return streams


def tile(streams, cols, blank):
    """Yield grid frames one by one. Finished cells freeze on their last frame."""
    rows = math.ceil(len(streams) / cols)
    cells = streams + [[]] * (rows * cols - len(streams))
    length = max(len(s) for s in cells)

    def frame_at(stream, t):
        if not stream:
            return blank
        return stream[t] if t < len(stream) else stream[-1]

    for t in range(length):
        yield np.vstack([
            np.hstack([frame_at(cells[r * cols + c], t) for c in range(cols)])
            for r in range(rows)
        ])


def pick_seeds(args):
    return args.seeds if args.seeds else list(random.sample(range(1, 1580), args.n))
    # return args.seeds if args.seeds else list(range(args.start, args.start + args.n))


# ---------------- Modes ----------------
def watch(weights, seeds):
    env = make_env("human")
    for seed in seeds:
        score, _ = play(env, weights, seed)
        print(f"Seed {seed:5d} | reward {score:7.1f} | {'LANDED' if score >= 200 else 'failed'}")
        time.sleep(1)
    env.close()


def record(weights, seeds, args):
    env = make_env("rgb_array")
    font = get_font(int(18 * args.scale) + 6)
    episodes, scores = [], []
    for seed in seeds:
        score, frames = play(env, weights, seed, record=True, frame_skip=args.frame_skip)
        label = f"seed {seed} | reward {score:.0f}"
        episodes.append([prepare(f, label, args.scale, font) for f in frames])
        scores.append(score)
        print(f"Seed {seed:5d} | reward {score:7.1f} | {len(frames)} frames")
    env.close()

    fps = ENV_FPS / args.frame_skip   # keeps real-time speed
    hold = int(fps * args.hold)        # pause after each landing

    # Without --grid it's just a single cell playing everything in sequence
    if args.grid:
        cols = args.cols
        rows = args.rows or math.ceil(len(episodes) / cols)
    else:
        cols, rows = 1, 1

    streams = build_streams(episodes, rows * cols, hold)
    blank = np.zeros_like(episodes[0][0])
    frames = tile(streams, cols, blank)

    if args.out.lower().endswith(".mp4"):
        # Streamed to disk frame by frame: much lighter on memory
        count = 0
        with imageio.get_writer(args.out, fps=fps) as writer:
            for f in frames:
                writer.append_data(f)
                count += 1
    else:
        frames = list(frames)
        count = len(frames)
        imageio.mimsave(args.out, frames, duration=1000 / fps, loop=0)

    landed = sum(s >= 200 for s in scores)
    print(f"Saved {args.out} | {rows}x{cols} grid | {count} frames | landed {landed}/{len(scores)}")


def test(weights, seeds):
    env = make_env()
    scores = np.array([play(env, weights, s)[0] for s in seeds])
    env.close()
    print(f"Test on {len(seeds)} seeds: {scores.mean():.1f} ± {scores.std():.1f}")
    print(f"Successful landings (>=200): {np.mean(scores >= 200):.0%}")
    print(f"Worst seed: {seeds[int(scores.argmin())]} ({scores.min():.1f}) | "
          f"Best seed: {seeds[int(scores.argmax())]} ({scores.max():.1f})")


def main():
    p = argparse.ArgumentParser(description="Watch, record or test the evolved lander.")
    p.add_argument("mode", choices=["watch", "record", "test"])
    p.add_argument("--weights", default="best_weights.npy")
    p.add_argument("--seeds", type=int, nargs="+", help="explicit list of seeds")
    p.add_argument("--start", type=int, default=1000, help="first seed if --seeds not given")
    p.add_argument("--n", type=int, default=3, help="number of seeds if --seeds not given")
    # recording options
    p.add_argument("--out", default="landing.gif", help=".gif or .mp4")
    p.add_argument("--grid", action="store_true", help="tile landings side by side")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--rows", type=int, default=None,
                   help="fix grid rows; extra landings queue up inside the cells")
    p.add_argument("--scale", type=float, default=0.5, help="frame scale (smaller = lighter file)")
    p.add_argument("--frame-skip", type=int, default=2, help="keep 1 of every N frames")
    p.add_argument("--hold", type=float, default=1.0, help="seconds to pause after each landing")
    args = p.parse_args()

    weights = np.load(args.weights)
    seeds = pick_seeds(args)

    if args.mode == "watch":
        watch(weights, seeds)
    elif args.mode == "record":
        record(weights, seeds, args)
    else:
        test(weights, seeds)


if __name__ == "__main__":
    main()