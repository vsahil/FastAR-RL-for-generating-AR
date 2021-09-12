#! /bin/bash
pip install -r requirements.txt
pip install -e fastcf/baselines
pip install -e fastcf/gym-midline
pysmt-install --z3 --confirm-agreement      # For MACE
