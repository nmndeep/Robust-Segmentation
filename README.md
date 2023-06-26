<h2>Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models </h2>

*Francesco Croce, Naman D Singh, Matthias Hein*

University of Tübingen


[Paper](https://arxiv.org/abs/2306.12941)	

`Abstract`	While a large amount of work has focused on designing adversarial attacks against image classifiers, only a few methods exist to attack semantic segmentation models. We show that attacking segmentation models presents task-specific challenges, for which we propose novel solutions. Our final evaluation protocol outperforms existing methods, and shows that those can overestimate the robustness of the models. Additionally, so far adversarial training, the most successful way for obtaining robust image classifiers, could not be successfully applied to semantic segmentation. We argue that this is because the task to be learned is more challenging, and requires significantly higher computational effort than for image classification. As a remedy, we show that by taking advantage of recent advances in robust ImageNet classifiers, one can train adversarially robust segmentation models at limited computational cost by fine-tuning robust backbones.

---------------------------------
<h3>Experimental setup and code</h3>

Main dependencies: `PyTorch-2.0.0, torchvision-0.15.0, timm-0.6.2, AutoAttack` 


<h4>Segmentation Ensemble Attack (SEA) evaluation</h4>

Run runner_infer.sh with the models config (.yaml) file from [configs](/configs) folder.

This computes the final adversarial robustness for the particular dataset and model passed as arguments within the `.yaml` file.
_________________________________
<h4>Training</h4>

SLURM type setup in `runner.sh` , run with `location_of_config` file and `num_of_gpu` as arguments.

For non-SLURM directly run [train.py](/tools/train.py) with `location_of_config` file and `num_of_gpu` as arguments.


- For `UperNet` with `ConvNext` (both Tiny and Small versions) backbone  for `ADE20K`

 	-  Clean-training: config-file: *ade20k_convnext_cvst.yaml* set `BACKBONE` with `CONVNEXT-S_CVST` for Small model. 
	-  Adversarial-training: config-file: *ade20k_convnext_rob_cvst.yaml* set `BACKBONE` with `CONVNEXT-S_CVST` for Small model.
 
- For `UperNet` with `ConvNext` (both Tiny and Small versions) backbone  for `PASCALVOC`
  
	-  Clean-training: config-file: *pascalvoc_convnext_cvst.yaml* set `BACKBONE` with `CONVNEXT-S_CVST` for Small model. 
	-  Adversarial-training: config-file: *pascalvoc_convnext_rob_cvst.yaml* set `BACKBONE` with `CONVNEXT-S_CVST` for Small model.
   
- For `SegMenter` with `Vit-S` backbone for `ADE20K` dataset
  
	-  Adversarial-training: config-file: *ade20k_segmenter_clean.yaml*, set `ADVERSARIAL` to FALSE for clean training.

_________________________________

<h4> Robust-Segmentation models</h4>

We make our robust models publically available. mIoU is reported for clean evaluation and with SEA evaluation at two perturbation strengths.
| model-name              | Dataset    | Clean(mIoU) | Adv.(4/255) | Adv.(8/255) |    checkpoint                             |
|-------------------------|------------|-------------|-------------|-------------|-------------------------------------------|
| UperNet-ConvNext-T_CVST | PASCAL-VOC |     75.2%    |     63.8%    |     37.0%    | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/zSFgoAngcm47FZm)     |
| UperNet-ConvNext-S_CVST | PASCAL-VOC |     76.6%    |     66.2%    |     38.0%    | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/MBXnMd5QKztmZaa)     |
| UperNet-ConvNext-T_CVST | ADE20K     |     31.7%    |     18.6%    |     6.70%    | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/ACMQRiyfyXboXwT)     |
| UperNet-ConvNext-S_CVST | ADE20K     |     32.1%    |     19.2%    |     7.20%    | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/Smogk2BWbfMxkyo)     |
Note: the models are trained including the background class for both VOC and ADE20K.
*SegMenter-ViT-S model available soon.

Robust pre-trained backbone models were taken from [Revisiting-AT](https://github.com/nmndeep/revisiting-at)* github repository.

*Note: For UperNet we always use the ConvNext backbone with Convolution Stem (CvSt).
_________________________________

<h4>Required citations</h4>

If you use our code/models consider citing us with the follwong BibTex entry:
```
@article{croce2023robust,
 title={Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models}, 
 author={Francesco Croce and Naman D Singh and Matthias Hein},
 year={2023},
 journal={arXiv:2306.12941}}
```

Also consider citing [SegPGD](https://arxiv.org/abs/2207.12391) if you use SEA attack, as their loss funciton makes up a part of SEA evaluation.

<h5>Acknowledgements</h5>

The code in this repo is partially based on the following publically available codebases.

-  [https://github.com/hszhao/semseg](https://github.com/hszhao/semseg)
-  [https://github.com/Tramac/awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
-  [https://github.com/rstrudel/segmenter)](https://github.com/rstrudel/segmenter)
-  [https://github.com/facebookresearch/ConvNeXt/tree/main/semantic_segmentation](https://github.com/facebookresearch/ConvNeXt/tree/main/semantic_segmentation)
-  [https://huggingface.co/docs/transformers/main/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation](https://huggingface.co/docs/transformers/main/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation)
