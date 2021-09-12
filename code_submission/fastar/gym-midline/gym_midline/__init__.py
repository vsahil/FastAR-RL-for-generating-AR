from gym.envs.registration import register


# German Credit
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

