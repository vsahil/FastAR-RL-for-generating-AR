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
    id='sine-v10000',
    entry_point='gym_midline.envs:FollowSine10000',
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



# German Credit - reduced features (6)

register(
    id='germanreduced-v01',
    entry_point='gym_midline.envs:GermanCreditReduced01',
)

register(
    id='germanreduced-v1',
    entry_point='gym_midline.envs:GermanCreditReduced1',
)

register(
    id='germanreduced-v10',
    entry_point='gym_midline.envs:GermanCreditReduced10',
)

register(
    id='germanreduced-v100',
    entry_point='gym_midline.envs:GermanCreditReduced100',
)

register(
    id='germanreduced-v1000',
    entry_point='gym_midline.envs:GermanCreditReduced1000',
)


# German Credit - all features
register(
    id='german-v0',
    entry_point='gym_midline.envs:GermanCredit0',
)

register(
    id='german-v01',
    entry_point='gym_midline.envs:GermanCredit01',
)

register(
    id='german-v1',
    entry_point='gym_midline.envs:GermanCredit1',
)

register(
    id='german-v10',
    entry_point='gym_midline.envs:GermanCredit10',
)

register(
    id='german-v100',
    entry_point='gym_midline.envs:GermanCredit100',
)

register(
    id='german-v1000',
    entry_point='gym_midline.envs:GermanCredit1000',
)


# Adult Income
register(
    id='adult-v0',
    entry_point='gym_midline.envs:AdultIncome0',
)

register(
    id='adult-v01',
    entry_point='gym_midline.envs:AdultIncome01',
)

register(
    id='adult-v1',
    entry_point='gym_midline.envs:AdultIncome1',
)

register(
    id='adult-v10',
    entry_point='gym_midline.envs:AdultIncome10',
)

register(
    id='adult-v100',
    entry_point='gym_midline.envs:AdultIncome100',
)

register(
    id='adult-v1000',
    entry_point='gym_midline.envs:AdultIncome1000',
)




# Credit Default
register(
    id='default-v0',
    entry_point='gym_midline.envs:CreditDefault0',
)

register(
    id='default-v01',
    entry_point='gym_midline.envs:CreditDefault01',
)

register(
    id='default-v1',
    entry_point='gym_midline.envs:CreditDefault1',
)

register(
    id='default-v10',
    entry_point='gym_midline.envs:CreditDefault10',
)

register(
    id='default-v100',
    entry_point='gym_midline.envs:CreditDefault100',
)

register(
    id='default-v1000',
    entry_point='gym_midline.envs:CreditDefault1000',
)
