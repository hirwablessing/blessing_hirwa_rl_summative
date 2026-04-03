from gymnasium.envs.registration import register

register(
    id="AfricanLiteracyTutor-v0",
    entry_point="environment.custom_env:AfricanLiteracyTutorEnv",
)
