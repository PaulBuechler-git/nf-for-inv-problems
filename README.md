# nf-for-inv-problems
Repository accompanying the Bachelor Thesis of Paul Büchler for the implementations of the image reconstruction task.
This are the most important scripts to run the training and experiments. The normalizing flow used in this implementations 
is based on the FrEIA library (https://github.com/vislearn/FrEIA) and the implementations from the PatchNr Regulariser paper
with a repository that can be found here (https://github.com/FabianAltekrueger/patchNR).

These files are the most important for 

### Training and Training evaluation
- train_patchNr_buterfly.py 
- train_patchNr_material.py
- train_patchNr_butterfly.ipynb
- train_patchNr_material.ipynb

### Prior Blur Experiments and evaluation
- prior_blurr_sensitivity_experiment_butterfly.ipynb
- prior_blurr_sensitivity_experiment_butterfly.py
- prior_blurr_sensitivity_experiment_comp.ipynb
- prior_blurr_sensitivity_experiment_material.ipynb
- prior_blurr_sensitivity_experiment_material.py

### Prior Noise Experiments and evaluation
- prior_noise_sensitivity_experiment_butterfly.ipynb
- prior_noise_sensitivity_experiment_butterfly.py
- prior_noise_sensitivity_experiment_comb.ipynb
- prior_noise_sensitivity_experiment_material.ipynb
- prior_noise_sensitivity_experiment_material.py

### Reconstruction and evaluation
- debluring.py
- deblurring.ipynb
- reconstruction_experiment_inc_weight_butterfly.ipynb
- reconstruction_experiment_inc_weight_butterfly.py
- reconstruction_experiment_inc_weight_material.ipynb
- reconstruction_experiment_inc_weight_material.py

