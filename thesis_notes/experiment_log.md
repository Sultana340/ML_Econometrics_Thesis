
## Baseline Experiment

## Experiment 1: Modified Baseline AGMM
-tau_fn: abs
-iv_strenth: 0.5
-num_data: 1000
-learner: default
Changes: Device solely CPU. 
Result:


Observation:

## Experiment 2: Modified Baseline AGMM
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



## How does AGMM perform across different structural function shapes, holding the rest of the design fixed?(Machine Learning focused Structural function complexity)
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

## Experiment 6: Run Started: 2026-04-20, 11:20 (Validation Experiment)
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

Linear case: easy inverse problem + smooth structure + near-ideal identification+ Best performance AGMM
Absolute value case: mid difficulty (non-smooth) + slightly higher MSE + estimator handles moderate non linearity+ Good performance(still strong R2)
Sin Case: high MSE + Neg R2 + High frequency variation + Instrument weakness + Finite sample issue

Interpretation: works extremely well for smooth/ low-complexity structure+ Stable under piecewise smooth function(abs) + But struggle with high frequency functions.

A Clean Narrative: AGMM perform strongly under smooth structural functions but deteriorates under highly nonlinear and oscillatory mappings, highlighting the interaction between function complexity, identification strength, and adversarial approximation.


## Experiment 7: Run Started: 2026-04-21, 11:20 (IV Strength Comparision)
Objective: To compare the performance of different IV strengths (low, medium, high).
Run Started: 2026-04-20, 10:20
Script:run_thesis_iv_strength_comparison_final.py
tau_fn = ['abs']
 iv_strength = [0.3,0.6,0.9]
 estimators = ['AGMM'] 
 dgps = ['z_image'] 
 num_datas = [2000]
 monte carlo = 5
 Epoch = 200


###################################################
## Final Experiment




 ## Experiment 8:  How effectively does AGMM recover the structural function under varying levels of identification and structural complexity?
 Objective: To compare the performance of different IV strengths (low, medium, high) in the AGMM framework and structural complexity.
 Run time: 12hours+
Run Started: 2026-04-23, 10:20
Script:run_thesis_iv_strength_tau_function_comparision.py
tau_fn = ['sin','abs']
 iv_strength = [0.3,0.6,0.9]
 estimators = ['AGMM'] 
 dgps = ['z_image'] 
 num_datas = [2000]
 monte carlo = 5
 Epoch = 200
 Result:


Observation:

 
 ## Experiment 9:  How does the finite-sample performance of AGMM change when high dimensionality enters through the instrument space versus the treatment space?
 Objective: To assess the finite-sample performance of AGMM change when high dimensionality enters through the instrument space versus the treatment space.
 Run Started: 2026-04-24, 13:00
Script:run_thesis_comparision.py
tau_fn = ['abs','sin']
 iv_strength = [0.6]
 estimators = ['AGMM'] 
 dgps = ['z_image','x_image'] 
 num_datas = [2000]
 monte carlo = 3
 Epoch = 200
Result:


Observation:



 ## Experiment 10: Do kernel-based methodological variants improve structural recovery or training stability relative to the baseline AGMM implementation in a controlled high dimensional IV setting?
 
 Run Started:
 Run Time:
 Script:
  tau_fn_list = ["abs", "sin"]
 iv_strength_list = [0.6]
dgps = ["z_image"]
estimators = ["AGMM", "KernelLayerMMDGMM", "CentroidMMDGMM"]
monte_carlo = 5
num_data = 2000
epochs = 200


Then combine the results.        

Result:


Observation:

## Experiment 11: How stable is the adversarial training procedure under practical computational constraints?






## Experiment 12: How does the proposed AGMM implementation relate to existing neural instrument-variable estimators, particularly DeepGMM and DeepIV, with respect to structural recovery, endogeneity control, and computational feasibility in nonparametric IV settings?

## agmm vs deepiv last check:
tau_fn_list = ["abs", "sin"]
iv_strength_list = [0.6]
dgps = ["z_image"]
estimators = ["AGMM", "DeepIV"]
monte_carlo = 5
num_data = 2000
## agmm vs deepgmm last check:
tau_fn_list = ["abs", "sin"]
iv_strength_list = [0.6]
dgps = ["z_image"]
estimators = ["AGMM", "DeepGMM"]
num_data_list = [2000]
monte_carlo = 5
epochs = 200
batch_size = 100
device = "cpu"




 