import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.font.init()

SCORE_FONT = pygame.font.SysFont("arial", 25)

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE_1 = (0, 0, 255)
BLUE_2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
FPS = 144


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")


class SnakeGameAI:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.window = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.reset()

        pygame.display.set_caption("Snake Game")

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = (
            random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE)
            * BLOCK_SIZE
        )
        y = (
            random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE)
            * BLOCK_SIZE
        )
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def _update_ui(self):
        self.window.fill(BLACK)

        for point in self.snake:
            pygame.draw.rect(
                self.window,
                BLUE_1,
                pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.window,
                BLUE_2,
                pygame.Rect(point.x + 4, point.y + 4, 12, 12),
            )

        pygame.draw.rect(
            self.window,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        score_text = SCORE_FONT.render(f"Score: {self.score}", True, WHITE)
        self.window.blit(score_text, (0, 0))

        pygame.display.update()

    def _move(self, action):
        # [straight, right, left]

        clockwise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]
        index = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clockwise[index]
        elif np.array_equal(action, [0, 1, 0]):
            new_direction = clockwise[(index + 1) % 4]
        else:
            new_direction = clockwise[(index - 1) % 4]

        self.direction = new_direction

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def is_collision(self, point=None):
        if point is None:
            point = self.head
        if (
            point.x > self.width - BLOCK_SIZE
            or point.x < 0
            or point.y > self.height - BLOCK_SIZE
            or point.y < 0
        ):
            return True

        if point in self.snake[1:]:
            return True

        return False

    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return (
                reward,
                game_over,
                self.score,
                self.snake,
                self.frame_iteration,
            )

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(FPS)

        return reward, game_over, self.score, self.snake, self.frame_iteration
