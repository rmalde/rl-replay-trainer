from dataclasses import dataclass

from replay_trainer.data.convert.utils import create_default_init_fn

@dataclass(init=False)
class GameConfig:
    gravity: float
    boost_consumption: float
    dodge_deadzone: float

    __slots__ = tuple(__annotations__)

    exec(create_default_init_fn(__slots__))