

## Experiment 1: Modified Baseline AGMM
-tau_fn: abs
-iv_strenth: 0.5
-num_data: 1000
-learner: default
Changes: Device solely CPU. 
Result:


Observation:

## Experiment 2: Baseline AGMM
-tau_fn: abs
-iv_strenth: 0.5
-num_data: 1000
-learner: default

Result:


Observation:


## Experiment 3: THESIS AGMM 
-tau_fn: abs
-iv_strenth: 0.5
-num_data: 1000
-learner: default

Result:


Observation:

## Experiment 4: THESIS AGMM(Num_data increased)
tau_fn = ['abs']
 iv_strength = [0.5]
 estimators = ['AGMM', 'KernelLayerMMDGMM'] 
 dgps = ['z_image'] 
 num_datas = [2000] 
Edit: Klayer_model has been deleted
Result:


Observation:



## How does AGMM perform across different structural function shapes, holding the rest of the design fixed?
## Experiment 5: Run Started: 2026-04-20, 10:20 (Verification Experiment)
Objective: the verification of the script and to make a standard basis for compare the performance of different tau functions (linear, sin, abs) in the AGMM framework.
Run Started: 2026-04-20, 10:20
Script:run_thesis_tau_comparision.py
tau_fn = ['linear','sin','abs']
 iv_strength = [0.5]
 estimators = ['AGMM'] 
 dgps = ['z_image'] 
 num_datas = [2000]
 monte carlo = 1
Changes in code: I've added the sin_fn function and updated the fn_dict in agmm_mnist_dgps.py to include 'sin'. The script should now be able to use the 'sin' tau function without errors. The sin_fn is defined as np.sin(x), which applies the sine function to the input.The linear function is already defined linear_fn(x) = 2 * x in agmm_mnist_dgps.py. Installed packages torch,torchaudio,torchvision.
Result:

## Experiment 6: Run Started: 2026-04-20, 10:20 (Validation Experiment)
Objective: To compare the performance of different tau functions (linear, sin, abs) in the AGMM framework.
Run Started: 2026-04-20, 10:20
Script:run_thesis_tau_comparision.py
tau_fn = ['linear','sin','abs']
 iv_strength = [0.5]
 estimators = ['AGMM'] 
 dgps = ['z_image'] 
 num_datas = [2000]
 monte carlo = 5
Changes in code: I've added the sin_fn function and updated the fn_dict in agmm_mnist_dgps.py to include 'sin'. The script should now be able to use the 'sin' tau function without errors. The sin_fn is defined as np.sin(x), which applies the sine function to the input.The linear function is already defined linear_fn(x) = 2 * x in agmm_mnist_dgps.py. Installed packages torch,torchaudio,torchvision.
Result: