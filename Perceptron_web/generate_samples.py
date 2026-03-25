import json
import random

random.seed(42)

GRID_SIZE = 4

def empty_grid():
    return [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

def flatten(grid):
    return [cell for row in grid for cell in row]

def add_noise(grid, max_flips=1):
    """
    Add very small controlled noise while trying to preserve the letter.
    """
    g = [row[:] for row in grid]
    flips = random.randint(0, max_flips)
    for _ in range(flips):
        r = random.randint(0, GRID_SIZE - 1)
        c = random.randint(0, GRID_SIZE - 1)
        g[r][c] = 1 if g[r][c] == 0 else 0
    return g

def make_L(col=0, height=4, foot_len=3, base_row=3):
    """
    Make an L:
    - vertical line at 'col'
    - bottom horizontal line at 'base_row'
    """
    g = empty_grid()

    start_row = max(0, base_row - height + 1)
    for r in range(start_row, base_row + 1):
        g[r][col] = 1

    for c in range(col, min(GRID_SIZE, col + foot_len)):
        g[base_row][c] = 1

    return g

def make_T(top_row=0, stem_col=1, bar_len=3, stem_len=4, start_col=None):
    """
    Make a T:
    - top horizontal bar
    - vertical stem downward
    """
    g = empty_grid()

    if start_col is None:
        start_col = max(0, stem_col - (bar_len // 2))
    end_col = min(GRID_SIZE, start_col + bar_len)

    for c in range(start_col, end_col):
        g[top_row][c] = 1

    end_row = min(GRID_SIZE, top_row + stem_len)
    for r in range(top_row, end_row):
        g[r][stem_col] = 1

    return g

def dataset_key(x):
    return tuple(x)

def generate_L_samples():
    samples = []

    # Many L variations possible on 4x4
    for col in range(0, 2):                 # left or slightly shifted right
        for height in range(2, 5):          # 2..4
            for foot_len in range(2, 5):    # 2..4
                for base_row in range(2, 4):# row 2 or 3
                    g = make_L(col, height, foot_len, base_row)
                    samples.append((flatten(g), 0))

                    # noisy versions
                    for _ in range(3):
                        noisy = add_noise(g, max_flips=1)
                        samples.append((flatten(noisy), 0))
    return samples

def generate_T_samples():
    samples = []

    for top_row in range(0, 2):             # row 0 or 1
        for stem_col in range(1, 3):        # col 1 or 2
            for bar_len in range(2, 5):     # 2..4
                for stem_len in range(2, 5):# 2..4
                    start_col = max(0, stem_col - (bar_len // 2))
                    g = make_T(top_row, stem_col, bar_len, stem_len, start_col)
                    samples.append((flatten(g), 1))

                    # noisy versions
                    for _ in range(3):
                        noisy = add_noise(g, max_flips=1)
                        samples.append((flatten(noisy), 1))
    return samples

def unique_samples(samples):
    seen = set()
    unique = []
    for x, y in samples:
        key = (tuple(x), y)
        if key not in seen:
            seen.add(key)
            unique.append((x, y))
    return unique

def main():
    L_samples = unique_samples(generate_L_samples())
    T_samples = unique_samples(generate_T_samples())

    random.shuffle(L_samples)
    random.shuffle(T_samples)

    # Make exactly 60 of each = 120 total
    L_samples = L_samples[:60]
    T_samples = T_samples[:60]

    dataset = []
    for x, y in L_samples + T_samples:
        dataset.append({"x": x, "y": y})

    X = [item["x"] for item in dataset]
    y = [item["y"] for item in dataset]

    print("Total samples:", len(dataset))
    print("L samples:", len(L_samples))
    print("T samples:", len(T_samples))

    print("\nCopy this into app.js as TRAINING_DATA:\n")
    print(json.dumps(dataset, indent=2))

if __name__ == "__main__":
    main()
