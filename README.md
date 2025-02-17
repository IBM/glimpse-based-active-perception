# Glimpse-based active perception

## Setting up the project environment
Install the required packages specified in [```requirements.txt```](requirements.txt) running
```
pip install -r requirements.txt
```

Next, configure the paths to the datasets and logs in [```definitions.py```](definitions.py).

## Datasets
All datasets except for ART must be generated based on instructions of the corresponding works. 

To generate the SVRT dataset use the following [link](https://fleuret.org/cgi-bin/gitweb/gitweb.cgi?p=pysvrt.git;a=summary) 
from the original work [1].
To generate OOD datasets for SVRT #1 (same-different) task follow the instructions in the
[repository](https://github.com/GuillermoPuebla/same_different_paper) of the 
original work [2]. To generate ART and CLEVR-ART dataset follow the instructions in the
[repository](https://github.com/Shanka123/OCRA) of the official work [3].
To generate our custom OOD test sets for CLEVR-ART, follow the same instructions as for generating 
the original CLEVR-ART dataset after having completed 2 additional steps: 1) move our blender templates in 
[```datasets/clevr_art_ood_blender_templates```](datasets/clevr_art_ood_blender_templates) to ```data/shapes``` in the 
cloned [repository](https://github.com/Shanka123/OCRA); 2) in that cloned repository, change line 380 
in ```render_images_<task_name>.py``` to ```object_mapping = [('Sphere', 'sphere'), ('SmoothPyramid', 'pyramid')]``` 
for the OOD-1 dataset and to ```object_mapping = [('SmoothTorus', 'torus'), ('SmoothPyramid', 'pyramid')]``` 
for the OOD-2 dataset.

## Running the experiments
Experiments on SVRT, SVRT#1-OOD, ART and CLEVR-ART are specified in files in the correspondingly named folders in 
[```./exeperiments```](experiments).
In each of those folders the scripts named ```glimpse_transformer.py``` correspond to models that use Transformer-based 
downstream architecture.
Scripts named ```glimpse_abs_*.py``` correspond to models with Abstractor-based downstream architecture with * standing 
for the sensor type.
Each scripts contain initialization of every important model's component so that one can easily adjust various 
parameters for experimenting.

To reproduce our results, you may need to adjust the inputs to the constructor of ```Exepriment``` class
under ```if __name__ == "__main__":``` specifying the seed,
sensor type and task name/index, if applicable.
To start an experiment for specific dataset and model, simply run the corresponding script. For example,
to start an experiment on SVRT using the model with the Abstractor and log-polar glimpse sensor run the following 
from the source directory of this project:
```
PYTHONPATH=. python experiments/svrt_sd_ood/glimpse_abs_log_polar.py
```
The accuracy results are saved as tensorboard logs and can be accessed by running:
```
tensorboard --logdir <path_to_logs>
```
where ```<path_to_logs>``` is specified in the ```super()``` call at the beginning of ```__init__()``` method of ```Experiment```
class. 

## References
The implementation of the Abstractor was taken from the
[official repository](https://github.com/slotabstractor/slotabstractor) of [4]

[1] Fleuret, F., Li, T., Dubout, C., Wampler, E. K., Yantis, S., & Geman, D. (2011). Comparing machines and humans on a visual categorization test. Proceedings of the National Academy of Sciences, 108(43), 17621–17625.

[2] Puebla, G., & Bowers, J. S. (2022). Can deep convolutional neural networks support relational reasoning in the same-different task? Journal of Vision, 22(10), 11. 

[3] Webb, T., Mondal, S. S., & Cohen, J. D. (2024). Systematic visual reasoning through object-centric relational abstraction. Advances in Neural Information Processing Systems, 36.

[4] Mondal, S.S., Cohen, J.D., & Webb, T.W. (2024). Slot Abstractors: Toward Scalable Abstract Visual Reasoning.
ArXiv, abs/2403.03458.



