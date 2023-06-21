<h3>Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models </h3>

Main dependencies: `PyTorch-2.0.0, torchvision-0.15.0, timm-0.6.2` 

Segmentation Ensemble Attack (SEA) evaluation/ Clean performance evaluation:
Run runner_infer.sh with the models config (.yaml) file from `configs` folder.
This computes the final adversarial robustness for the particular dataset and model passed as arguments within the .yaml file.


Training
SLURM type setup in `runner.sh` with `location_of_config` file and `num_of_gpu` as arguments
- For `UperNet` with `ConvNext` (both Tiny and Small versions) backbone  for `ADE20K`
	-  Clean-training: config-file: ade20k_convnext_cvst.yaml replace `BACKBONE` with ''CONVNEXT-S_CVST'' for Small model 
	-  Adversarial-training: config-file: ade20k_convnext_rob_cvst.yaml replace `BACKBONE` with ''CONVNEXT-S_CVST'' for Small model 
- For `UperNet` with `ConvNext` (both Tiny and Small versions) backbone  for `PASCALVOC`
	-  Clean-training: config-file: pascalvoc_convnext_cvst.yaml replace `BACKBONE` with ''CONVNEXT-S_CVST'' for Small model 
	-  Adversarial-training: config-file: pascalvoc_convnext_rob_cvst.yaml replace `BACKBONE` with ''CONVNEXT-S_CVST'' for Small model 
- `SegMenter` with `Vit-S` backbone for `ADE20K` dataset.
	-  Adversarial-training: config-file: ade20k_segmenter_clean, set `ADVERSARIAL` to FALSE for clean training.
