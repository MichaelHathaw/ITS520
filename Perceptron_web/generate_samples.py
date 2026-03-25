import json

GRID = 4

def empty_grid():
    return [[0 for _ in range(GRID)] for _ in range(GRID)]

def flatten(grid):
    return [cell for row in grid for cell in row]

def grid_key(grid):
    return tuple(flatten(grid))

def make_T(top_row, left_col, bar_width, stem_height, stem_offset):
    """
    T:
    - top horizontal bar of width 2..4
    - vertical stem of height 2..4
    - stem can be under ANY bar cell
    """
    g = empty_grid()

    # top bar
    for c in range(left_col, left_col + bar_width):
        g[top_row][c] = 1

    # stem
    stem_col = left_col + stem_offset
    for r in range(top_row, top_row + stem_height):
        g[r][stem_col] = 1

    return g

def make_L(top_row, left_col, height, foot_width):
    """
    L:
    - vertical line of height 2..4
    - bottom foot of width 2..4 extending right
    """
    g = empty_grid()

    # vertical line
    for r in range(top_row, top_row + height):
        g[r][left_col] = 1

    # bottom foot
    bottom_row = top_row + height - 1
    for c in range(left_col, left_col + foot_width):
        g[bottom_row][c] = 1

    return g

def generate_T_samples():
    samples = {}
    for bar_width in range(2, GRID + 1):
        for stem_height in range(2, GRID + 1):
            for stem_offset in range(bar_width):  # stem can be under any bar cell
                for top_row in range(0, GRID - stem_height + 1):
                    for left_col in range(0, GRID - bar_width + 1):
                        g = make_T(top_row, left_col, bar_width, stem_height, stem_offset)
                        samples[grid_key(g)] = {
                            "x": flatten(g),
                            "y": 1
                        }
    return list(samples.values())

def generate_L_samples():
    samples = {}
    for height in range(2, GRID + 1):
        for foot_width in range(2, GRID + 1):
            for top_row in range(0, GRID - height + 1):
                for left_col in range(0, GRID - foot_width + 1):
                    g = make_L(top_row, left_col, height, foot_width)
                    samples[grid_key(g)] = {
                        "x": flatten(g),
                        "y": 0
                    }
    return list(samples.values())

def main():
    t_samples = generate_T_samples()
    l_samples = generate_L_samples()
    dataset = t_samples + l_samples

    print("Total T samples:", len(t_samples))
    print("Total L samples:", len(l_samples))
    print("Total dataset:", len(dataset))

    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("Saved dataset.json")

if __name__ == "__main__":
    main()
