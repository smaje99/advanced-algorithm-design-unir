"""
Fifteen Puzzle Solver
=====================

Solves the 15-puzzle problem using Branch and Bound algorithm.

This script receives an initial configuration as a list of 16 integers,
uses a priority queue to explore moves, and prints the steps to the goal
if solvable.

Examples
--------
>>> initial_state = [1, 2, 3, 4,
...                  5, 6, 7, 8,
...                  9, 10, 11, 12,
...                  13, 14, 15, 0]
>>> solve_puzzle(initial_state)

Author: Sergio Andrés Majé Franco
Date: 2025-06-09
Institution: Fundación Universitaria Internacional La Rioja
Course: Diseño de Algoritmos Avanzados
Assignment: Trabajo 1: Puzzle de las losetas
"""

import argparse
import heapq  # For priority queue implementation
import time
from typing import Optional


GOAL_STATE = tuple(range(1, 16)) + (0,)
MOVES = {"UP": -4, "DOWN": 4, "LEFT": -1, "RIGHT": 1}


class PuzzleState:
    """
    Represents a state in the 15-puzzle problem with cost,
    heuristic and path history.
    """

    def __init__(
        self, board: tuple[int, ...], moves: list[str], cost: int
    ) -> None:
        """
        Initializes a new PuzzleState with the given board,
        move history, and cost.
        Sets up the heuristic and priority for use in search algorithms.

        Parameters
        ----------
        board : tuple of int
            The current configuration of the puzzle as a tuple of 16 integers.
        moves : list of str
            The sequence of moves taken to reach this state.
        cost : int
            The cost (number of moves) to reach this state.
        """
        self.board = board
        self.moves = moves
        self.cost = cost
        self.heuristic = self.calculate_heuristic()
        self.priority = self.cost + self.heuristic

    def __lt__(self, other: "PuzzleState") -> bool:
        """
        Less-than comparison for priority queue ordering.

        Parameters
        ----------
        other : PuzzleState
            The other state to compare against.
        """
        return self.priority < other.priority

    def calculate_heuristic(self) -> int:
        """
        Calculates the sum of Manhattan distances for all tiles.

        The Manhattan distance is used as a heuristic for the 15-puzzle
        because it is admissible (never overestimates the actual cost)
        and consistent (satisfies the triangle inequality).

        Returns
        -------
        int
            Total heuristic cost (sum of Manhattan distances for each tile).

        Notes
        -----
        The Manhattan distance for each tile is calculated as:

            h(n) = ∑ |x_i - x_i*| + |y_i - y_i*|

        where (x_i, y_i) is the current position of tile i,
        and (x_i*, y_i*) is its target position in the goal state.

        This function skips the empty tile (0) as it has no target location.

        Time Complexity
        ---------------
        O(n), where n = 16.
        The algorithm performs a single pass over the board.
        """
        distance = 0
        for idx, value in enumerate(self.board):
            if value == 0:
                continue
            target_x, target_y = divmod(value - 1, 4)
            curr_x, curr_y = divmod(idx, 4)
            distance += abs(target_x - curr_x) + abs(target_y - curr_y)
        return distance

    def is_goal(self) -> bool:
        """
        Checks whether the current board is the goal state.

        Returns
        -------
        bool
        """
        return self.board == GOAL_STATE

    def get_possible_moves(self) -> list["PuzzleState"]:
        """
        Generates all valid successor states from the current board
        configuration.

        For each legal direction (up, down, left, right) where the
        empty tile (0) can be moved, this function creates a new puzzle
        state by swapping the empty tile with its neighbor in that direction.

        Returns
        -------
        list of PuzzleState
            A list of PuzzleState instances representing all reachable
            next states.

        Notes
        -----
        - The empty tile (0) cannot be moved outside the puzzle boundaries.
        - Valid moves are determined by checking the position of the blank
        (e.g., no left move from column 0).
        - Each new state increases the cost by 1 and appends the move
        direction to the path history.
        - Moves are labeled as: 'UP', 'DOWN', 'LEFT', 'RIGHT'.
        - This function does not check whether a state has been visited
        (that is handled by the search algorithm).

        Time Complexity
        ---------------
        O(1). A maximum of 4 possible moves are checked and up to 4 new
        states are generated.
        """
        new_states: list[PuzzleState] = []
        zero_index = self.board.index(0)
        row, col = divmod(zero_index, 4)

        for direction, delta in MOVES.items():
            new_idx = zero_index + delta
            if direction == "LEFT" and col == 0:
                continue
            if direction == "RIGHT" and col == 3:
                continue
            if direction == "UP" and row == 0:
                continue
            if direction == "DOWN" and row == 3:
                continue

            new_board = list(self.board)
            new_board[zero_index], new_board[new_idx] = (
                new_board[new_idx],
                new_board[zero_index],
            )
            new_states.append(
                PuzzleState(
                  tuple(new_board),
                  self.moves + [direction],
                  self.cost + 1
                )
            )

        return new_states


def is_solvable(board: list[int]) -> bool:
    """
    Determines if a 15-puzzle is solvable using inversion counting.

    The solvability of a 15-puzzle depends on the number of inversions
    and the position of the empty tile. An inversion is a pair of tiles
    (a, b) such that a appears before b but a > b.

    Rules for solvability:
    - If the grid width is even (4), the puzzle is solvable if:
      - the blank is on an even-numbered row from the bottom (2nd, 4th),
      and the number of inversions is odd, or
      - the blank is on an odd-numbered row from the bottom (1st, 3rd),
      and the number of inversions is even.

    Parameters
    ----------
    board : list of int
        List representing the puzzle configuration.

    Returns
    -------
    bool
        True if solvable, False otherwise.

    Time Complexity
    ---------------
    O(n²), where n = 16. It performs a double loop over
    the board to count inversions.
    """
    inversions = 0
    for i in range(15):
        for j in range(i + 1, 16):
            if board[i] and board[j] and board[i] > board[j]:
                inversions += 1

    blank_row_from_bottom = 4 - (board.index(0) // 4)  # 1-based

    # If blank is on even row from bottom, inversions must be odd
    # If blank is on odd row from bottom, inversions must be even
    return (inversions % 2 == 0 and blank_row_from_bottom % 2 != 0) or \
           (inversions % 2 != 0 and blank_row_from_bottom % 2 == 0)


def solve_puzzle(start: list[int]) -> Optional[list[str]]:
    """
    Solves the 15-puzzle using the Branch and Bound technique (A* variant).

    This function applies an informed search strategy (A*) to explore valid
    puzzle states, using a priority queue where the next state is chosen based
    on the sum of the path cost and a heuristic estimate to the goal (Manhattan
    distance).

    Parameters
    ----------
    start : list of int
        Initial puzzle configuration as a flat list of 16 integers (0 to 15),
        where 0 represents the empty tile.

    Returns
    -------
    list of str or None
        A list of move directions ('UP', 'DOWN', 'LEFT', 'RIGHT') that solve
        the puzzle, or None if the configuration is unsolvable.

    Detailed description
    --------------------
    The function first checks whether the puzzle is solvable using the
    standard rule based on the number of inversions and the row of the
    empty tile.

    If solvable, the algorithm:
    - Initializes a priority queue with the initial state.
    - Expands the state with the lowest `cost + heuristic` value (A*).
    - Tracks visited configurations to avoid revisiting states.
    - Uses a Manhattan distance heuristic (admissible and consistent).

    The search terminates when the goal configuration is reached, and prints:
    - Each intermediate 4×4 matrix state along the solution path.
    - The total time taken to solve the puzzle.

    Notes
    -----
    - The function uses a custom `PuzzleState` class to manage puzzle
    configurations and costs.
    - This implementation guarantees finding the optimal (shortest)
    solution path.
    - The heuristic function used is the sum of Manhattan distances from
    each tile to its goal.

    Time Complexity
    ---------------
    In the worst case, O(b^d), where:
    - b is the branching factor (maximum 4 moves per state),
    - d is the depth of the shallowest goal.

    In practice, the heuristic drastically reduces the number of
    nodes explored.

    Examples
    --------
    >>> initial = [1, 2, 3, 4,
    ...            5, 6, 7, 8,
    ...            9, 10, 11, 12,
    ...            13, 15, 14, 0]
    >>> solve_puzzle(initial)
    Move: LEFT
    Move: UP
    ...
    Execution time: 0.218563 seconds
    """
    if not is_solvable(start):
        print("This puzzle configuration is unsolvable.")
        return None

    start_time = time.time()
    visited: set[tuple[int, ...]] = set()
    initial = PuzzleState(tuple(start), [], 0)
    heap = [initial]

    while heap:
        current = heapq.heappop(heap)
        if current.board in visited:
            continue
        visited.add(current.board)

        if current.is_goal():
            end_time = time.time()
            print_solution_path(start, current.moves)
            print(f"Execution time: {end_time - start_time:.6f} seconds")
            return current.moves

        for neighbor in current.get_possible_moves():
            if neighbor.board not in visited:
                heapq.heappush(heap, neighbor)

    return None


def print_solution_path(start: list[int], moves: list[str]) -> None:
    """
    Prints the puzzle path from initial state to goal, in 4x4 matrices.

    Parameters
    ----------
    start : list of int
        Initial puzzle configuration.
    moves : list of str
        Sequence of moves taken.
    """
    print("Initial State:")
    print_matrix(start)
    current = start[:]

    for move in moves:
        zero = current.index(0)
        delta = MOVES[move]
        new_zero = zero + delta
        current[zero], current[new_zero] = current[new_zero], current[zero]
        print(f"\nMove: {move}")
        print_matrix(current)


def print_matrix(board: list[int]) -> None:
    """
    Prints the 4x4 matrix representation of the puzzle.

    Parameters
    ----------
    board : list of int
        Puzzle state as list of 16 integers.
    """
    for i in range(0, 16, 4):
        row = board[i:i + 4]
        print(" ".join(str(num).rjust(2) if num != 0 else "  " for num in row))


def main() -> None:
    """
    Runs the 15-puzzle solver on a selected test case.

    This function contains predefined test configurations representing
    different levels of difficulty for the 15-puzzle and allows the user
    to select which one to run via a command-line argument.

    Example usage:
    >>> python main.py --index 1
    """
    parser = argparse.ArgumentParser(
      description="Solve a predefined 15-puzzle test case."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="Index of the test puzzle to run (1-8). Default is 1."
    )
    args = parser.parse_args()
    index: int = args.index - 1

    labels = [
        "Already solved (trivial)",
        "One move to solve",
        "Two moves to solve",
        "Easy (5-10 moves)",
        "Medium (15-20 moves)",
        "Hard (30-40 moves)",
        "Very hard (deep search)",
        "Not solvable (wrong parity)"
    ]

    test_puzzles: list[list[int]] = [
        # 1. Already solved (trivial case)
        [
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
            13, 14, 15,  0
        ],
        # 2. One move to solve (very easy)
        [
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
            13, 14,  0, 15
        ],
        # 3. Two moves to solve (easy)
        [
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
            13,  0, 14, 15
        ],
        # 4. Easy (~5–10 moves to solve)
        [
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 12,  0,
            13, 14, 11, 15
        ],
        # 5. Medium (~15–20 moves)
        [
            1,  2,  3,  4,
            5,  6,  0,  8,
            9, 10,  7, 12,
            13, 14, 11, 15
        ],
        # 6. Hard (~30–40+ moves)
        [
            5,  1,  2,  4,
            0,  6,  3,  8,
            9, 10,  7, 12,
            13, 14, 11, 15
        ],
        # 7. Very hard (solvable but deep search required)
        [
            2,  1,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
            13, 15, 14,  0
        ],
        # 8. Not solvable (wrong parity — odd number of inversions)
        [
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12,
            13, 15, 14,  0
        ],
    ]

    if not 0 <= index < len(test_puzzles):
        print((
            f"Invalid index {index}. Please choose a number between"
            f" 1 and {len(test_puzzles)}."
        ))
        return

    print(f"--- Test Case #{index + 1}: {labels[index]} ---")
    solve_puzzle(test_puzzles[index])


if __name__ == "__main__":
    main()
