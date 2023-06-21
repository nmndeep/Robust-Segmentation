<h3>Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models </h3>

Main dependencies: `PyTorch-2.0.0, torchvision-0.15.0, timm-0.6.2` 


<h4>Segmentation Ensemble Attack (SEA) evaluation</h4>

Run runner_infer.sh with the models config (.yaml) file from [configs](/configs) folder.

This computes the final adversarial robustness for the particular dataset and model passed as arguments within the .yaml file.

<h4>Training</h4>

SLURM type setup in `runner.sh` with `location_of_config` file and `num_of_gpu` as arguments
- For `UperNet` with `ConvNext` (both Tiny and Small versions) backbone  for `ADE20K`

 	-  Clean-training: config-file: *ade20k_convnext_cvst.yaml* set `BACKBONE` with `CONVNEXT-S_CVST` for Small model 
	-  Adversarial-training: config-file: *ade20k_convnext_rob_cvst.yaml* set `BACKBONE` with `CONVNEXT-S_CVST` for Small model
 
- For `UperNet` with `ConvNext` (both Tiny and Small versions) backbone  for `PASCALVOC`
  
	-  Clean-training: config-file: *pascalvoc_convnext_cvst.yaml* set `BACKBONE` with `CONVNEXT-S_CVST` for Small model 
	-  Adversarial-training: config-file: *pascalvoc_convnext_rob_cvst.yaml* set `BACKBONE` with `CONVNEXT-S_CVST` for Small model
   
- `SegMenter` with `Vit-S` backbone for `ADE20K` dataset.
  
	-  Adversarial-training: config-file: *ade20k_segmenter_clean.yaml*, set `ADVERSARIAL` to FALSE for clean training.

<h4> Robust-Segmentation models</h4>

We make our robust models publically available. 
| model-name              | Dataset    | checkpoint                                                                  |
|-------------------------|------------|-----------------------------------------------------------------------------|
| UperNet-ConvNext-T_CVST | PASCAL-VOC | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/zSFgoAngcm47FZm)     |
| UperNet-ConvNext-S_CVST | PASCAL-VOC | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/MBXnMd5QKztmZaa)     |
| UperNet-ConvNext-T_CVST | ADE20K     | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/ACMQRiyfyXboXwT)     |
| UperNet-ConvNext-S_CVST | ADE20K     | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/Smogk2BWbfMxkyo)     |
*SegMenter model available soon.

Robust pre-trained backbone models were taken from [Revisiting-AT](https://github.com/nmndeep/revisiting-at) github repository.

<h5>Acknowledgements</h5>

The code in this repo is partially based on the following publically available codebases

 [1](https://github.com/hszhao/semseg), [2](https://github.com/Tramac/awesome-semantic-segmentation-pytorch), [3](https://github.com/rstrudel/segmenter/),  [4](https://github.com/facebookresearch/ConvNeXt/tree/main/semantic_segmentation), and [5](https://huggingface.co/docs/transformers/main/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation)
