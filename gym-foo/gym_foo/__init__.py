from gym.envs.registration import register

register(
    id='spacex-v0',
    entry_point='gym_foo.envs:RocketLander',
)