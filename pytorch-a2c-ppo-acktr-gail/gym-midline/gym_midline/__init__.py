from gym.envs.registration import register

register(
    id='midline-v0',
    entry_point='gym_midline.envs:ReachMidLine',
)


register(
    id='step-v01',
    entry_point='gym_midline.envs:FollowStep01',
)

register(
    id='step-v1',
    entry_point='gym_midline.envs:FollowStep1',
)

register(
    id='step-v10',
    entry_point='gym_midline.envs:FollowStep10',
)

register(
    id='step-v100',
    entry_point='gym_midline.envs:FollowStep100',
)

register(
    id='step-v1000',
    entry_point='gym_midline.envs:FollowStep1000',
)


register(
    id='sine-v01',
    entry_point='gym_midline.envs:FollowSine01',
)

register(
    id='sine-v1',
    entry_point='gym_midline.envs:FollowSine1',
)

register(
    id='sine-v10',
    entry_point='gym_midline.envs:FollowSine10',
)

register(
    id='sine-v100',
    entry_point='gym_midline.envs:FollowSine100',
)

register(
    id='sine-v1000',
    entry_point='gym_midline.envs:FollowSine1000',
)


register(
    id='trapezium-v01',
    entry_point='gym_midline.envs:FollowTrapezium01',
)

register(
    id='trapezium-v1',
    entry_point='gym_midline.envs:FollowTrapezium1',
)

register(
    id='trapezium-v10',
    entry_point='gym_midline.envs:FollowTrapezium10',
)

register(
    id='trapezium-v100',
    entry_point='gym_midline.envs:FollowTrapezium100',
)

register(
    id='trapezium-v1000',
    entry_point='gym_midline.envs:FollowTrapezium1000',
)
