# From: https://github.com/lucas-emery/rocket-league-gym/blob/v2/rlgym/rocket_league/api/car.py

import numpy as np

from dataclasses import dataclass
from typing import Optional, Generic

from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM
from rlgym_sim.utils.gamestates import PhysicsObject

from replay_trainer.data.convert.utils import AgentID, DOUBLEJUMP_MAX_DELAY, FLIP_TORQUE_TIME, create_default_init_fn

@dataclass(init=False)
class Car(Generic[AgentID]):

    # Misc Data
    team_num: int  #TODO switch to typed class?
    hitbox_type: int  # TODO should probably be typed too?
    ball_touches: int  # number of ball touches since last state was sent
    bump_victim_id: Optional[AgentID]

    # Actual State
    demo_respawn_timer: float  # 0 if alive
    # TODO add num_wheels_contact when it's available in rsim
    #num_wheels_contact: int  # Needed for stuff like AutoRoll and some steering shenanigans
    on_ground: bool  # this is just numWheelsContact >=3 TODO make property when num_w_cts is available
    supersonic_time: float  # greater than 0 when supersonic, needed for state set since ssonic threshold changes with time
    boost_amount: float
    boost_active_time: float  # you're forced to boost for at least 12 ticks
    handbrake: float

    # Jump Stuff
    has_jumped: bool
    is_holding_jump: bool  # whether you pressed jump last tick or not
    is_jumping: bool  # changes to false after max jump time
    jump_time: float  # need jump time for state set, doesn't reset to 0 because of psyonix's landing jump cooldown

    # Flip Stuff
    has_flipped: bool
    has_double_jumped: bool
    air_time_since_jump: float
    flip_time: float
    flip_torque: np.ndarray

    # AutoFlip Stuff - What helps you recover from turtling
    is_autoflipping: bool
    autoflip_timer: float
    autoflip_direction: float  # 1 or -1, determines roll direction

    physics: PhysicsObject
    _inverted_physics: PhysicsObject

    __slots__ = tuple(__annotations__)

    exec(create_default_init_fn(__slots__))

    @property
    def is_blue(self) -> bool:
        return self.team_num == BLUE_TEAM

    @property
    def is_orange(self) -> bool:
        return self.team_num == ORANGE_TEAM

    @property
    def is_demoed(self) -> bool:
        return self.demo_respawn_timer > 0

    @property
    def is_boosting(self) -> bool:
        return self.boost_active_time > 0

    @property
    def is_supersonic(self) -> bool:
        return self.supersonic_time > 0

    @property
    def can_flip(self) -> bool:
        return not self.has_double_jumped and not self.has_flipped and self.air_time_since_jump < DOUBLEJUMP_MAX_DELAY

    @property  # TODO This one isn't in rsim python yet, emulate with prop
    def is_flipping(self) -> bool:
        return self.has_flipped and self.flip_time < FLIP_TORQUE_TIME

    @is_flipping.setter
    def is_flipping(self, value: bool):
        if value:
            self.has_flipped = True
            if self.flip_time >= FLIP_TORQUE_TIME:
                self.flip_time = 0
        else:
            self.flip_time = FLIP_TORQUE_TIME

    @property
    def had_car_contact(self) -> bool:
        return self.bump_victim_id is not None

    @property
    def inverted_physics(self) -> PhysicsObject:
        if self._inverted_physics is None:
            self._inverted_physics = self.physics.inverted()
        return self._inverted_physics