import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set


@dataclass
class Point:
    x: int
    y: int


class Player:
    def __init__(self, x: int, y: int):
        self.head = Point(x, y)
        self.tail: List[Point] = [Point(x, y)]  # Keep using tail
        self.alive = True

    def move(self, dx: int, dy: int) -> None:
        """Move the player and leave a tail trail"""
        self.head.x += dx
        self.head.y += dy
        self.tail.append(Point(self.head.x, self.head.y))

    def get_head_tuple(self) -> Tuple[int, int]:
        return (self.head.x, self.head.y)

    def get_tail_tuples(self) -> List[Tuple[int, int]]:
        return [(p.x, p.y) for p in self.tail]


class TronEnvironment:
    def __init__(self, num_players: int = 4):
        self.grid_width = 30
        self.grid_height = 20
        self.num_players = num_players
        self.players: Dict[int, Player] = {}
        self.current_player_index = 0  # Index of current player in alive players list
        self.reset()

    def reset(self) -> None:
        self.players = {}
        positions = set()

        for player_id in range(self.num_players):
            while True:
                x = random.randint(0, self.grid_width - 1)
                y = random.randint(0, self.grid_height - 1)
                if (x, y) not in positions:
                    break

            self.players[player_id] = Player(x, y)
            positions.add((x, y))

        self.alive_players = list(self.players.keys())

    def kill_player(self, player_id: int):
        """Mark player as dead and clear their tail"""
        self.players[player_id].alive = False
        self.players[player_id].tail = []
        if player_id in self.alive_players:
            self.alive_players.remove(player_id)

    def get_game_state(self) -> List[str]:
        state = []
        for player_id in range(self.num_players):
            player = self.players[player_id]
            if not player.alive:
                state.append("-1 -1 -1 -1")
            else:
                tail_start = player.tail[0] if player.tail else player.head
                state.append(f"{tail_start.x} {tail_start.y} {player.head.x} {player.head.y}")
        return state

    def execute_move(self, player_id: int, direction: str) -> bool:
        player = self.players[player_id]
        if not player.alive:
            return False

        dx, dy = 0, 0
        if direction == "UP":
            dy = -1
        elif direction == "DOWN":
            dy = 1
        elif direction == "LEFT":
            dx = -1
        elif direction == "RIGHT":
            dx = 1

        new_x, new_y = player.head.x + dx, player.head.y + dy

        # Check boundary collision
        if new_x < 0 or new_x >= self.grid_width or new_y < 0 or new_y >= self.grid_height:
            self.kill_player(player_id)
            return False

        # Precompute all obstacle positions
        obstacles: Set[Tuple[int, int]] = set()
        for pid, p in self.players.items():
            if p.alive:
                obstacles.update(p.get_tail_tuples())
                if pid != player_id:  # Don't check own head
                    obstacles.add(p.get_head_tuple())

        # Check collision
        if (new_x, new_y) in obstacles:
            self.kill_player(player_id)
            return False

        player.move(dx, dy)
        return True

    def run_turn(self, move: str) -> Optional[List[str]]:
        """Process current player's turn and automatically switch to next player"""
        if not self.alive_players:
            return None

        current_player = self.alive_players[self.current_player_index % len(self.alive_players)]

        if not self.execute_move(current_player, move):
            if self.is_game_over():
                return None
            return self.get_game_state()

        # Update to next player
        self.current_player_index = (self.current_player_index + 1) % len(self.alive_players)
        return self.get_game_state()

    def is_game_over(self) -> bool:
        return len(self.alive_players) <= 1

    def get_winner(self) -> Optional[int]:
        return self.alive_players[0] if len(self.alive_players) == 1 else None

    def get_obs(self, player_id: int) -> List[float]:
        player = self.players[player_id]
        if not player.alive:
            return [0.0] * (8 + 8 + 24)

        # 1. 8-direction distances (8 dimensions)
        direction_obs = self._get_direction_distances(player)

        # 2. Normalized coordinates (8 dimensions)
        position_obs = self._get_normalized_positions(player_id)

        # 3. 5x5 global pooling (24 dimensions)
        global_obs = self._get_global_pooling_observation()

        return direction_obs + position_obs + global_obs

    def _get_direction_distances(self, player: Player) -> List[float]:
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        distances = []
        for dx, dy in directions:
            dist = self._ray_cast(player.head.x, player.head.y, dx, dy)
            distances.append(1.0 / (dist + 0.01))
        return distances

    def _ray_cast(self, x: int, y: int, dx: int, dy: int) -> int:
        distance = 0
        while True:
            x += dx
            y += dy
            distance += 1

            if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
                return distance

            for pid, other_player in self.players.items():
                if not other_player.alive:
                    continue
                if (x, y) == other_player.get_head_tuple():
                    return distance
                if (x, y) in other_player.get_tail_tuples():
                    return distance

    def _get_normalized_positions(self, current_player_id: int) -> List[float]:
        positions = []
        current_player = self.players[current_player_id]

        positions.extend([
            current_player.head.x / self.grid_width,
            current_player.head.y / self.grid_height
        ])

        opponent_count = 0
        for pid, player in self.players.items():
            if pid != current_player_id and player.alive and opponent_count < 3:
                positions.extend([
                    player.head.x / self.grid_width,
                    player.head.y / self.grid_height
                ])
                opponent_count += 1

        while opponent_count < 3:
            positions.extend([-1.0, -1.0])
            opponent_count += 1

        return positions

    def _get_global_pooling_observation(self) -> List[float]:
        obs = []
        region_w, region_h = 5, 5

        obstacles = set()
        for player in self.players.values():
            if player.alive:
                obstacles.add(player.get_head_tuple())
                obstacles.update(player.get_tail_tuples())

        for ry in range(0, self.grid_height, region_h):
            for rx in range(0, self.grid_width, region_w):
                count = 0
                for y in range(ry, min(ry + region_h, self.grid_height)):
                    for x in range(rx, min(rx + region_w, self.grid_width)):
                        if (x, y) in obstacles:
                            count += 1
                obs.append(count / (region_w * region_h))
        return obs


if __name__ == "__main__":
    env = TronEnvironment(num_players=4)

    while not env.is_game_over():
        print(f"{env.num_players} {env.current_player_index}")
        for line in env.get_game_state():
            print(line)

        valid_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        ai_move = random.choice(valid_moves)
        print(ai_move)

        env.run_turn(ai_move)  # change current player

    winner = env.get_winner()
    print(f"Game over! Winner: Player {winner}" if winner else "All players crashed!")