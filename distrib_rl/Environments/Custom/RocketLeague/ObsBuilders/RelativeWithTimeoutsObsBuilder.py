import math
import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates.physics_object import PhysicsObject
from .DefaultWithTimeoutsObsBuilder import DefaultWithTimeoutsObsBuilder

class RelativeWithTimeoutsObsBuilder(DefaultWithTimeoutsObsBuilder):
    def __init__(self, 
            pad_teams_to=3,
            obs_per_player=44,
            pos_coef=1/2300, 
            ang_coef=1/math.pi, 
            lin_vel_coef=1/2300, 
            ang_vel_coef=1/math.pi, 
            tick_skip=8
        ):
        """
        :param pad_teams_to: Number of teammates to pad model to
        :param obs_per_player: Number of zeros to pad observation with per player
        :param pos_coef: Position normalization coefficient
        :param ang_coef: Rotation angle normalization coefficient
        :param lin_vel_coef: Linear velocity normalization coefficient
        :param ang_vel_coef: Angular velocity normalization coefficient
        :param tick_skip: how many ticks are skipped between updates - used for
                          determining how much game time has transpired
        """
        
        self.pad_teams_to = pad_teams_to
        self.obs_per_player = obs_per_player

        super().__init__(pos_coef, ang_coef, lin_vel_coef, ang_vel_coef, tick_skip)


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


        allies, enemies = [], []
        ally_count, enemy_count = 0, 0

        for i, other in enumerate(state.players):

            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
                ally_count += 1
                if ally_count > (self.pad_teams_to - 1):
                    # Arbitrarily ignore agents over tean pad size
                    continue
            else:
                team_obs = enemies
                enemy_count += 1
                if enemy_count > self.pad_teams_to:
                    # Arbitrarily ignore agents over tean pad size
                    continue

            self._add_player_to_obs(team_obs, other, i, inverted)
            self._add_ball_relative(team_obs, other, ball, inverted)
            self._add_player_relative(team_obs, player, other, inverted)

        obs.extend(allies)
        ally_deficit = self.pad_teams_to - ally_count - 1
        if ally_deficit > 0:
            obs.extend( [ [0.0]*self.obs_per_player*ally_deficit ] )

        obs.extend(enemies)
        enemy_deficit = self.pad_teams_to - enemy_count
        if enemy_deficit > 0:
            obs.extend( [ [0.0]*self.obs_per_player*enemy_deficit ] )

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
