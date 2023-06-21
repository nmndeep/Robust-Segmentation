<h3>Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models </h3>

Main dependencies: `PyTorch-2.0.0, torchvision-0.15.0, timm-0.6.2` 

Segmentation Ensemble Attack (SEA) evaluation/ Clean performance evaluation:
Run runner_infer.sh with the models config (.yaml) file from `configs` folder.
This computes the final adversarial robustness for the particular dataset and model passed as arguments within the .yaml file.


Training
SLURM type setup in `runner.sh` with `location_of_config` file and `num_of_gpu` as arguments
- For `UperNet` with `ConvNext` (both Tiny and Small versions) backbone  for `ADE20K` 
- For `UperNet` with `ConvNext` (both Tiny and Small versions) backbone  for `PASCALVOC` - pass.
- `SegMenter` with `Vit-S` backbone for `ADE20K` dataset.
- UperNet code adapted from Huggingface transformers UperNetforSegmentation.
- ConvNext code taken from the official repo.


Pass the respective config file in `configs` folder as argument.
TRAINING:
- `run_train.sh` runs on GPU VMs
- `runner.sh` on SLURM.
- `train.py` has the main training script.


EVALUATION FOR SEA PIPELINE:
	- Replace 'MODEL_PATH' in EVAL in the respective config file with the trained model.
Run either (with config file passed as a argument).
	- `infer.py` does evaluation on standard 1.4k images of PASCALVOC.   OR
	- 'infer_ade.py' does eval on ADE20K dataset.

- `attack_type`: set to `apgd-larg-eps` for SEA - run with `pair`, 0,1,2,3 then pass the location of the files to 'strr' in worse_case_miou.py.
- Run 'tools/worse_case_miou.py' for final SEA numbers.
