# Snake game
from collections import deque
from enum import Enum

import numpy as np


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


DIRECTION_2_IDX = {d: i for i, d in enumerate(Direction)}


class SnakeGame:
    def __init__(self, h: int, w: int) -> None:
        self.height = h
        self.width = w
        # Start in the middle facing up
        self.body = deque([(h // 2 + 1, w // 2), (h // 2, w // 2)])
        self.body_set = set(self.body)
        self.direction = Direction.UP
        self.food: tuple[int, int]
        self.spawn_food()
        self.moves_without_food = 0

    @property
    def state(self) -> np.ndarray:
        # Channels: Body, food
        res: np.ndarray = np.zeros(
            (2, self.height, self.width), dtype=np.float32
        )
        res[0][tuple(zip(*self.body))] = np.linspace(
            0.1, 1, len(self.body), dtype=np.float32
        )
        res[1][self.food] = 1
        res = res.flatten()

        # Additional feature: our heading
        heading = np.zeros(4, dtype=np.float32)
        heading[DIRECTION_2_IDX[self.direction]] = 1

        return np.concatenate((res, heading))

    def __str__(self) -> str:
        delta2str = {
            (0, 2): "-",
            (0, -2): "-",
            (2, 0): "|",
            (-2, 0): "|",
            (-1, 1): "/",
            (-1, -1): "\\",
            (1, 1): "\\",
            (1, -1): "/",
        }
        grid = [[" "] * self.width for _ in range(self.height)]
        for i in range(1, len(self.body) - 1):
            prev, curr, nex = self.body[i - 1], self.body[i], self.body[i + 1]
            delta = nex[0] - prev[0], nex[1] - prev[1]
            grid[curr[0]][curr[1]] = delta2str[delta]

        grid[self.body[0][0]][self.body[0][1]] = "*"
        grid[self.body[-1][0]][self.body[-1][1]] = "H"
        grid[self.food[0]][self.food[1]] = "F"

        # Pad grid with `#` to show border
        grid = [["#"] + r + ["#"] for r in grid]
        grid = [["#"] * (self.width + 2)] + grid + [["#"] * (self.width + 2)]
        return "\n".join(" ".join(r) for r in grid)

    def spawn_food(self) -> None:
        loc = (
            np.random.randint(0, self.height),
            np.random.randint(0, self.width),
        )
        while loc in self.body_set:
            loc = (
                np.random.randint(0, self.height),
                np.random.randint(0, self.width),
            )
        self.food = loc

    def move(self, direction: Direction) -> tuple[bool, float]:
        # Return (continue, reward)
        # Opposite direction = don't change
        if (
            self.direction.value[0] + direction.value[0],
            self.direction.value[1] + direction.value[1],
        ) != (0, 0):
            self.direction = direction

        # Move in direction
        head = self.body[-1]
        new = (
            head[0] + self.direction.value[0],
            head[1] + self.direction.value[1],
        )
        # Out of bounds check
        if (
            new[0] < 0
            or new[0] >= self.height
            or new[1] < 0
            or new[1] >= self.width
        ):
            return False, -1

        # Move body
        tail = self.body.popleft()
        self.body_set.remove(tail)

        # Crash body check needs to be after the tail moves out
        if new in self.body:
            return False, -1

        # And put the new head in
        self.body.append(new)
        self.body_set.add(new)

        # Food get check
        if new == self.food:
            self.moves_without_food = 0
            # Undo cutting the tail
            self.body.appendleft(tail)
            self.body_set.add(tail)

            # Spawn new food
            if len(self.body) == self.width * self.height:
                # Game is over, snake won
                return False, 10
            self.spawn_food()
            return True, 1
        else:
            self.moves_without_food += 1
            if self.moves_without_food >= self.width * self.height * 2:
                return False, 0
            return True, 0


class SnakeGameCNN(SnakeGame):
    def __init__(self, h: int, w: int) -> None:
        super().__init__(h, w)
        self.prev_state_no_history = np.zeros_like(
            self.state_no_history, dtype=np.float32
        )

    @property
    def state(self) -> np.ndarray:
        return np.concatenate(
            (self.state_no_history, self.prev_state_no_history),
            axis=0,
        )

    @property
    def state_no_history(self) -> np.ndarray:
        # Channels: head, body, food, prev head, prev body, prev food
        res: np.ndarray = np.zeros(
            (3, self.height, self.width), dtype=np.float32
        )
        res[0][self.body[-1]] = 1
        res[1][tuple(zip(*self.body))] = np.linspace(
            0.1, 1, len(self.body), dtype=np.float32
        )
        res[2][self.food] = 1

        return res

    def move(self, direction: Direction) -> tuple[bool, float]:
        self.prev_state_no_history = self.state_no_history
        return super().move(direction)


def human_play() -> None:
    inps = "udlr"
    mapping = {c: d for c, d in zip(inps, Direction)}

    game = SnakeGameCNN(8, 8)
    cont = True
    while cont:
        print(game.state)
        print(game.state.shape)
        print(game)
        inp = input()
        try:
            cont, reward = game.move(mapping[inp])
        except KeyError:
            continue

        print(reward)


if __name__ == "__main__":
    human_play()
