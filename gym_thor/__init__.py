from gym.envs.registration import register

register(
    id='thor-v0',
    entry_point='gym_thor.envs:ThorEnv',
)
