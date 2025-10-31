# 🧩 Jane Street Puzzle Solutions

[![Python](https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Last commit](https://img.shields.io/github/last-commit/arkanemystic/janestreetpuzzles?style=for-the-badge)](https://github.com/arkanemystic/janestreetpuzzles/commits/main)
[![Repo size](https://img.shields.io/github/repo-size/arkanemystic/janestreetpuzzles?style=for-the-badge)](https://github.com/arkanemystic/janestreetpuzzles)

> My personal repository for solutions to the monthly puzzles from [Jane Street](https://www.janestreet.com/puzzles/).

---

## Solutions

Here is a list of the puzzles I've solved or attempted.

| Puzzle / Month           | Solution File                                                                                                                                                                                                                                                                    | Status     | Notes                                                                                                                                |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| **Hooks** / **[September 2025]** | [hooks11-9/hooks11working.py](https://github.com/arkanemystic/janestreetpuzzles/blob/main/hooks11-9/hooks11working.py)                       | ✅ Solved  | Solved using Constraint Programming (`cpmpy`, `numpy`, `scipy`).     |
| **Hooks** / **[September 2025]** | [hooks11-9/hooks11-improved.py](https://github.com/arkanemystic/janestreetpuzzles/blob/main/hooks11-9/hooks11-improved.py)                     | ✅ Solved  | Solved using backtracking search with Dancing Links (DLX) algorithm implementation. |
| **Hooks** / **[September 2025]** | [hooks11-9/hooks11-improvedv2.py](https://github.com/arkanemystic/janestreetpuzzles/blob/main/hooks11-9/hooks11-improvedv2.py)                 | ✅ Solved  | Solved using an SMT solver (`z3-solver`, `numpy`, `scipy`).       |
| **Robot Baseball** / **[October 2025]** | [robotbaseball.py](https://github.com/arkanemystic/janestreetpuzzles/blob/main/robotbaseball.py)                                                 | ✅ Solved | Symbolic mathematics (`sympy`) and numerical optimization (`scipy`, `numpy`) approach. |
| *More to come* | ...                                                                                                                                                                                                                                                                              | ...        |                                                                                                                                      |

## Dependencies

Solutions are written in Python 3. Common dependencies include:

* `numpy`
* `scipy`
* `sympy` (for Robot Baseball)
* `cpmpy` (for one Hooks solution)
* `z3-solver` (for one Hooks solution)
* `codetiming` (used for timing in some scripts)

Some directories may contain specific `requirements.txt` files (e.g., `hooks11-9/requirements.txt`), though these might not list all necessary libraries for *all* solution variants (like `cpmpy` or `z3-solver`). Install missing libraries using pip as needed (e.g., `pip install cpmpy z3-solver`).

## How to Run

Navigate to the directory containing the solution file and run it using Python:

```bash
# Example for a Hooks solution
python hooks11-9/hooks11working.py

# Example for Robot Baseball
python robotbaseball.py
````

## Example Output

Below is an image showing the output from one of the puzzle solvers (`hooks11working.py`).

<p align="center"\>
<img src="https://github.com/arkanemystic/janestreetpuzzles/blob/main/readmeExample.png?raw=true" alt="Example Solution Output" width="300"/\>
<img src"https://github.com/arkanemystic/janestreetpuzzles/blob/main/readmeExample1.png" alt="Example Solution Output" width="300">
</p\>
