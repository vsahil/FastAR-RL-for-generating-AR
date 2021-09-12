# Replication code for FastAR submission.

The directory structure is as follows:

```tree
root
├── fastar
├── baselines
```

The [fastar](fastar) directory contains the FastAR code and the [baselines](baselines) directory contains the code for all the baselines. 

## FastAR

### Setup Instructions

1. Create a virtual environment by running: `virtualenv -p python3.6 supplementary`. 
2. Activate the virtual environment by running: `source supplementary/bin/activate`. 
3. Install the requirements by running the command `bash install_requirements.sh`. This install the requirements and the gym environments. 

### Training and evaluating the agents. 

Commands for training and evaluating the agents for the three datasets) with specific hyper-parameters can be found [here](fastar/commands_to_run.md). For convenience, we provide the trained agents for the three datasets [here](fastar/output/trained_models/). The evaluation commands will load the trained agents and run them. 
The expected directory structure is: 

```tree
fastar
├── output
│   ├── trained_models
│   ├── results
```

### FastAR results

After running the evaluation command, the results will be printed in CSV format in [this](fastar/output/results) directory. 


## Baselines

### Random and Greedy baselines

For running the random and greedy baselines, 
```bash
cd baselines
python random_strategy.py $dataset
python greedy_strategy.py $dataset
```

The options for the dataset are "german", "adult" and "default", which correspond to the German Credit, Adult Income, and Credit Default datasets respectively. 
The computed metrics are saved in file [all_metrics_baselines.csv](baselines/results/all_metrics_baselines.csv). 

### DiCE-Random, DiCE-Genetic, and DiCE-KDTree baselines
All the supporting code for DiCE-based baselines is in the [dice_ml](baselines/dice_ml) directory. 
For the model-agnostic versions of DiCE specifically, we provide the code in the file [dice_model_agnostic](baselines/dice_model_agnostic.py). 
For running the approaches, use this command:

```bash
python dice_model_agnostic.py $dataset $model_agnostic_approach
```
where $dataset is one of "german", "adult" or "default", and model-agnostic approaches are "random", "genetic", and "kdtree". 
Similar to the last case, the computed metrics are saved in file [all_metrics_baselines.csv](baselines/results/all_metrics_baselines.csv). 


### DiCE-Gradient 
For DiCE-Gradient, we provide the code in the file [dice_gradient.py](baselines/dice_gradient.py). For running this approach, use this command:

```bash
python dice_gradient.py $dataset 
```
where $dataset is one of "german", "adult" or "default". Similar to the last case, the computed metrics are saved in file [all_metrics_baselines.csv](baselines/results/all_metrics_baselines.csv). 

### DiCE-VAE

Unlike other DiCE baselines, DiCE-VAE takes several hyper-parameters. We ran a hyperparamter exploration (using the file [hyperparam_dicevae.py](baselines/hyperparam_dicevae.py) ) and found the best working ones. For using DiCE-VAE, use this command:

1. For the German Credit dataset, use this command:
```bash
python dice_vae.py --dataset_name=german --epochs=25 --batch_size=64 --encoded_size=10 --lr=0.01 --validity_reg=40
```
2. For the Adult Income dataset, use this command:
```bash
python dice_vae.py --dataset_name=adult --epochs=25 --batch_size=1024 --encoded_size=50 --lr=0.001 --validity_reg=80
```
3. For the Credit Default dataset, use this command:
```bash
python dice_vae.py --dataset_name=default --epochs=25 --batch_size=2048 --encoded_size=30 --lr=0.05 --validity_reg=60
```

Similar to the last case, the computed metrics are saved in file [all_metrics_baselines.csv](baselines/results/all_metrics_baselines.csv). 

### MACE

For using the MACE tool, change directory into the [mace-master](baselines/mace-master) directory and run the following command. 

```bash
python batchTest.py -d german_our -m forest -n one_norm -a MACE_eps_1e-3 -b 0 -s 500
python batchTest.py -d german_our -m lr -n one_norm -a MACE_eps_1e-3 -b 0 -s 500
```
Recall that we ran MACE only for the German Credit dataset, and with logisic regression (LR) 
and random forest (RF) as the classifiers. 

The computed explanations from MACE are saved in the following directories:
1. [LR](baselines/mace-master/_experiments/2021.05.25_13.44.19__german_our__lr__one_norm__MACE_eps_1e-3__batch0__samples500__pid0)
2. [RF](baselines/mace-master/_experiments/2021.05.25_13.45.10__german_our__forest__one_norm__MACE_eps_1e-3__batch0__samples500__pid0)

The file [process_results.py](baselines/mace-master/process_results.py) was used to parse the results from MACE and calculate the evaluation metrics.
For parsing the results, use this command:
```bash
python process_MACE_results.py LR
python process_MACE_results.py RF
```

The metrics are saved in the usual file [all_metrics_baselines.csv](baselines/results/all_metrics_baselines.csv). 

