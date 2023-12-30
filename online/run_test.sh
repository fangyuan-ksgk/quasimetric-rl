# Test run script for online experiments
# exec python -m online.main env.kind=gcrl "${@}"
# Added agent.actor=null to test the environment -- discrete action forbids actor optimization ...
exec python -m online.little_main env.kind='minigrid_env' env.name='MiniGrid-FourRooms-v0' agent.actor=null