import torch
import pygame
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
from plotter import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        point_right = Point(head.x + 20, head.y)
        point_left = Point(head.x - 20, head.y)
        point_up = Point(head.x, head.y - 20)
        point_down = Point(head.x, head.y + 20)

        direction_right = game.direction == Direction.RIGHT
        direction_left = game.direction == Direction.LEFT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        # fmt: off
        state = [
            (direction_right and game.is_collision(point_right)) or
            (direction_left and game.is_collision(point_left)) or
            (direction_up and game.is_collision(point_up)) or
            (direction_down and game.is_collision(point_down)),

            (direction_right and game.is_collision(point_down)) or
            (direction_left and game.is_collision(point_up)) or
            (direction_up and game.is_collision(point_right)) or
            (direction_down and game.is_collision(point_left)),

            (direction_right and game.is_collision(point_up)) or
            (direction_left and game.is_collision(point_down)) or
            (direction_up and game.is_collision(point_left)) or
            (direction_down and game.is_collision(point_right)),

            direction_right,
            direction_left,
            direction_up,
            direction_down,
            
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(
            states, actions, rewards, next_states, game_overs
        )

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        self.epsilon = 80 - self.num_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_lengths = []
    plot_mean_lengths = []
    plot_time_steps = []
    plot_mean_time_steps = []
    total_score = 0
    max_score = 0
    max_length = 0
    max_time_steps = 0
    total_length = 0
    total_time_steps = 0
    agent = Agent()
    game = SnakeGameAI()

    try:
        while True:
            state_old = agent.get_state(game)

            final_move = agent.get_action(state_old)

            reward, game_over, score, snake, time_steps = game.play_step(
                final_move
            )
            state_new = agent.get_state(game)

            agent.train_short_memory(
                state_old, final_move, reward, state_new, game_over
            )

            agent.remember(state_old, final_move, reward, state_new, game_over)

            if game_over:
                game.reset()
                agent.num_games += 1
                agent.train_long_memory()

                if score > max_score:
                    max_score = score

                if len(snake) - 1 > max_length:
                    max_length = len(snake) - 1

                if time_steps > max_time_steps:
                    max_time_steps = time_steps

                print(
                    f"Game #{agent.num_games} - Score: {score}, Highest Score: {max_score} | "
                    f"Length: {len(snake) - 1}, Highest Length: {max_length} | "
                    f"Time Steps: {time_steps}, Highest Time Steps: {max_time_steps}\n"
                )

                plot_scores.append(score)
                total_score += score

                mean_score = total_score / agent.num_games
                plot_mean_scores.append(mean_score)

                plot_lengths.append(len(snake) - 1)
                total_length += len(snake) - 1

                mean_length = total_length / agent.num_games
                plot_mean_lengths.append(mean_length)

                plot_time_steps.append(time_steps)
                total_time_steps += time_steps

                mean_time_steps = total_time_steps / agent.num_games
                plot_mean_time_steps.append(mean_time_steps)

                plot(
                    plot_scores,
                    plot_mean_scores,
                    plot_lengths,
                    plot_mean_lengths,
                    plot_time_steps,
                    plot_mean_time_steps,
                )
    except pygame.error:
        return


if __name__ == "__main__":
    train()
