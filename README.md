<h3>Some code for semantic segmentation.</h3>

Dependencies: PyTorch-2.0.0, torchvision-0.15.0 

- Tested for `UperNet` with `ConvNext` backbone  for `ADE20K` dataset and `PASCALVOC`.
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