import math
import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates.physics_object import PhysicsObject
from .DefaultWithTimeoutsObsBuilder import DefaultWithTimeoutsObsBuilder

LARGE_BOOST_MASK = np.array([
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
    0.0
])

class RelativeWithTimeoutsObsBuilder(DefaultWithTimeoutsObsBuilder):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if (state != self._state):
            self._step_state(state)
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
            pad_timers = self.inverted_boost_pad_timers
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads
            pad_timers = self.boost_pad_timers

        obs = [ball.position * self.POS_COEF,
               ball.linear_velocity * self.LIN_VEL_COEF,
               ball.angular_velocity * self.ANG_VEL_COEF,
               previous_action, pads, pad_timers]

        player_index = state.players.index(player)
        self._add_player_to_obs(obs, player, player_index, inverted)
        self._add_ball_relative(obs, player, ball, inverted)

        allies = []
        enemies = []

        for i, other in enumerate(state.players):
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            self._add_player_to_obs(team_obs, other, i, inverted)
            self._add_ball_relative(team_obs, other, ball, inverted)
            self._add_player_relative(team_obs, player, other, inverted)

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_ball_relative(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        obs.extend([
            (player_car.position - ball.position) * self.POS_COEF,
            (player_car.linear_velocity - ball.linear_velocity) * self.LIN_VEL_COEF,
            (player_car.angular_velocity - ball.angular_velocity) * self.ANG_VEL_COEF,
        ])

        return player_car

    def _add_player_relative(self, obs: List, player: PlayerData, other: PlayerData, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
            other_car = other.inverted_car_data
        else:
            player_car = player.car_data
            other_car = other.inverted_car_data

        obs.extend([
            (other_car.position - player_car.position) * self.POS_COEF,
            (other_car.linear_velocity - player_car.linear_velocity) * self.LIN_VEL_COEF,
            other_car.forward() - player_car.forward(),
            other_car.up() - player_car.forward(),
            (other_car.angular_velocity - player_car.angular_velocity) * self.ANG_VEL_COEF,
        ])

        return player_car
