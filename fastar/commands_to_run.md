# German Credit
Train command: 

```bash 
python -W ignore main.py --algo ppo --use-gae --lr 0.001 --clip-param 0.2 --value-loss-coef 0.5 --num-processes 1 --num-steps 256 --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './output/trained_models/german5_sampletrain_search_01_0.99_256_0.001_0.2' --env-name 'gym_midline:german-v01' --num-env-steps 5000000 --save-interval 5000 --gamma 0.99
```

Evaluation command: 

```bash 
python -W ignore main.py --algo ppo --use-gae --lr 0.001 --clip-param 0.2 --value-loss-coef 0.5 --num-processes 1 --num-steps 256 --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './output/trained_models/german5_sampletrain_search_01_0.99_256_0.001_0.2' --env-name 'gym_midline:german-v01' --num-env-steps 5000000 --save-interval 5000 --gamma 0.99 --eval
```


# Adult Income
Train command: 

```bash 
python -W ignore main.py --algo ppo --use-gae --lr 0.0001 --clip-param 0.2 --value-loss-coef 0.5 --num-processes 1 --num-steps 256 --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './output/trained_models/adult_sampletrain_search_01_0.99_256_0.0001_0.2' --env-name 'gym_midline:adult-v01' --num-env-steps 10000000 --save-interval 5000 --gamma 0.99
```

Evaluation command: 

```bash 
python -W ignore main.py --algo ppo --use-gae --lr 0.0001 --clip-param 0.2 --value-loss-coef 0.5 --num-processes 1 --num-steps 256 --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './output/trained_models/adult_sampletrain_search_01_0.99_256_0.0001_0.2' --env-name 'gym_midline:adult-v01' --num-env-steps 10000000 --save-interval 5000 --gamma 0.99 --eval
```


# Credit Default
Train command: 

```bash 
python -W ignore main.py --algo ppo --use-gae --lr 0.001 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 256 --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './output/trained_models/default_sampletrain_search_01_0.99_256_0.001_0.1' --env-name 'gym_midline:default-v01' --num-env-steps 10000000 --save-interval 5000 --gamma 0.99
```

Evaluation command: 

```bash 
python -W ignore main.py --algo ppo --use-gae --lr 0.001 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 256 --num-mini-batch 4 --log-interval 50 --use-linear-lr-decay --entropy-coef 0.01 --save-dir './output/trained_models/default_sampletrain_search_01_0.99_256_0.001_0.1' --env-name 'gym_midline:default-v01' --num-env-steps 10000000 --save-interval 5000 --gamma 0.99 --eval
```
