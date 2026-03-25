import json

GRID = 4

def empty_grid():
    return [[0 for _ in range(GRID)] for _ in range(GRID)]

def flatten(grid):
    return [cell for row in grid for cell in row]

def print_grid(grid):
    for row in grid:
        print("".join(str(x) for x in row))
    print()

def grid_to_tuple(grid):
    return tuple(tuple(row) for row in grid)

def make_L(top_row, left_col, height, width):
    """
    L shape:
    - vertical line goes downward from (top_row, left_col)
    - bottom foot goes right from the bottom of that vertical line
    """
    g = empty_grid()

    # vertical part
    for r in range(top_row, top_row + height):
        g[r][left_col] = 1

    # bottom horizontal foot
    bottom_row = top_row + height - 1
    for c in range(left_col, left_col + width):
        g[bottom_row][c] = 1

    return g

def make_T(top_row, left_col, bar_width, stem_height, stem_col_offset):
    """
    T shape:
    - top horizontal bar starts at (top_row, left_col) with length bar_width
    - stem starts on the same top row and goes downward
    - stem_col_offset says where the stem is under the bar
      e.g. if left_col=1 and stem_col_offset=2, stem is at col 3
    """
    g = empty_grid()

    # top bar
    for c in range(left_col, left_col + bar_width):
        g[top_row][c] = 1

    stem_col = left_col + stem_col_offset

    # vertical stem
    for r in range(top_row, top_row + stem_height):
        g[r][stem_col] = 1

    return g

def generate_all_Ls():
    """
    Generate every possible L in a 4x4 grid:
    - height: 2..4
    - width: 2..4
    - placed anywhere it fits
    """
    unique = {}
    samples = []

    for height in range(2, GRID + 1):
        for width in range(2, GRID + 1):
            for top_row in range(0, GRID - height + 1):
                for left_col in range(0, GRID - width + 1):
                    g = make_L(top_row, left_col, height, width)
                    key = grid_to_tuple(g)
                    if key not in unique:
                        unique[key] = True
                        samples.append({
                            "x": flatten(g),
                            "y": 0,   # L = 0
                            "shape": "L"
                        })
    return samples

def generate_all_Ts():
    """
    Generate every possible T in a 4x4 grid:
    - bar_width: 2..4
    - stem_height: 2..4
    - stem can be at any position under the top bar
    - placed anywhere it fits
    """
    unique = {}
    samples = []

    for bar_width in range(2, GRID + 1):
        for stem_height in range(2, GRID + 1):
            for stem_col_offset in range(bar_width):
                for top_row in range(0, GRID - stem_height + 1):
                    for left_col in range(0, GRID - bar_width + 1):
                        g = make_T(top_row, left_col, bar_width, stem_height, stem_col_offset)
                        key = grid_to_tuple(g)
                        if key not in unique:
                            unique[key] = True
                            samples.append({
                                "x": flatten(g),
                                "y": 1,   # T = 1
                                "shape": "T"
                            })
    return samples

def verify_examples():
    """
    Verify the two patterns you mentioned are recognized as Ts.
    """
    ex1 = [
        [0,0,0,0],
        [0,1,1,1],
        [0,0,1,0],
        [0,0,1,0]
    ]

    ex2 = [
        [0,0,0,0],
        [0,0,0,0],
        [0,1,1,1],
        [0,0,1,0]
    ]

    t_samples = generate_all_Ts()
    t_set = {tuple(s["x"]) for s in t_samples}

    print("Example 1 is T:", tuple(flatten(ex1)) in t_set)
    print("Example 2 is T:", tuple(flatten(ex2)) in t_set)

def main():
    Ls = generate_all_Ls()
    Ts = generate_all_Ts()

    dataset = Ls + Ts

    print("Total L samples:", len(Ls))
    print("Total T samples:", len(Ts))
    print("Total dataset size:", len(dataset))
    print()

    verify_examples()

    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("\nSaved dataset to dataset.json")
    print("\nFirst 10 samples:")
    print(json.dumps(dataset[:10], indent=2))

if __name__ == "__main__":
    main()
