# RoReg Benchmark

## Preprocessing (backbone extraction)

The first thing to do is to use `run_preprocessing_global.py` to do the preprocessing using FCGF.

## RoReg run

Use `run_benchamrk_global.py` to run RoReg on the preprocessed data

## Evaluate

Use `evaluate_all.py` to create the results folder with the results files


## Evaluation on the "original" (RoReg paper) ETH experiments

Downalod the ETH dataset from `https://drive.google.com/file/d/1hyurp5EOzvWGFB0kOl5Qylx1xGelpxaQ/view?usp=sharing`.
Place the downloaded ETH data in the data folder like this:

```
data/
├── origin_data/
    └── ETH/
```

Run the preprocessing with `python3 testset.py`

Then run the registration and evaluation with `python3 Test.py --RD --RM --ET yohoo --keynum 1000 --testset ETH --tau_2 0.2 --tau_3 0.5 --ransac_ird 0.5`