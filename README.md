<div align="center">

<h3>Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models</h3>

**Francesco Croce\*, Naman D Singh\*, and Matthias Hein**

ECCV 2024

Paper: Coming soon

<h4>Abstract</h4>
</div>

Adversarial robustness has been studied  extensively in image classification, especially for the Linf-threat model, but significantly less so for related tasks such as object detection and semantic segmentation, where attacks turn out to be a much harder optimization problem than for image classification. We propose several problem-specific novel attacks minimizing different metrics in 
accuracy and mIoU. The ensemble of our attacks, SEA, shows that existing attacks severely overestimate the robustness of semantic segmentation models.
Surprisingly, existing attempts of adversarial training for semantic segmentation models turn out to be weak or even completely non-robust. We investigate why previous adaptations of adversarial training to semantic segmentation failed and  show how recently proposed robust ImageNet backbones can be used to obtain adversarially robust semantic segmentation models with up to six times less training time for PASCAL-VOC and the more challenging ADE20K.

---------------------------------
<p align="center"><img src="/assets/teaser.png" width="700"></p>

<div align="center">
<h3> Robust Semantic Segmentation models</h3>
</div>

Models trained via the PIR-AT scheme. mIoU is reported for clean evaluation and with SEA evaluation (Adv.) at two perturbation strengths.
<div align="center">
	
| Model name              | Dataset    | Clean | Adv.(4/255) | Adv.(8/255) |    Checkpoint                             |
|-------------------------|------------|-------------|-------------|-------------|-------------------------------------------|
| UperNet-ConvNext-T_CVST | PASCAL-VOC |     75.2%    |     64.9%    |     34.6%    | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/zSFgoAngcm47FZm)     |
| UperNet-ConvNext-S_CVST | PASCAL-VOC |     76.6%    |     66.0%    |     36.4%    | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/MBXnMd5QKztmZaa)     |
| UperNet-ConvNext-T_CVST | ADE20K     |     31.7%    |     17.2%    |     4.90%    | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/ACMQRiyfyXboXwT)     |
| UperNet-ConvNext-S_CVST | ADE20K     |     32.1%    |     17.9%    |     5.40%    | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/Smogk2BWbfMxkyo)     |
| Segmenter-ViT-S 	  | ADE20K     |     28.7%    |     14.9%    |     5.30%    | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/XF6Woa9G3eiGPig)     |
-------------------------------------------------------------------------------------------------
</div>
	Note: the models are trained including the background class for both VOC and ADE20K.

Robust pre-trained backbone models were taken from [Revisiting-AT](https://github.com/nmndeep/revisiting-at) repository.

For UperNet we always use the ConvNext backbone with Convolution Stem (CvSt).

------------------------------------------------------------


<h3>Segmentation Ensemble Attack (SEA)</h3>

SEA containes three complementary attacks

- Mask-Cross-Entropy-Balanced (Mask-ce-bal)

- Mask-Cross-Entropy (Mask-ce) 

- Jenson-Shannon Divergence (JS-Avg)
  
The attacks are run sequentially and then:

1. For `aACC`, image-wise worst case over the attacks is taken.

2. For `mIoU`, the worst case is computed by updating the running-mIoU image/attack-wise.

<h4>To run SEA evaluation</h4>

Run [run_infer.sh](run_infer.sh) with the models config (.yaml) file from [configs](/configs) folder.

This computes the worst-case `mIoU` and `aACC` after SEA attack for the particular dataset and model passed as arguments within the `.yaml` file.

Note: All dataset locations, modelNames, pretrained-model checkpoint paths are set in respective `config-file`.

All required packages can be found in `requirements.txt`
_________________________________
<h3> PIR-AT Training</h3>

SLURM type setup in [run_train_slurm.sh](run_train_slurm.sh), run with location of `config-file` and `num_of_gpu` as arguments.

For non-SLURM multi-GPU setup run [run_train.sh](run_train.sh) with location of `config-file` and `num_of_gpu` as arguments.

- For `UperNet` with `ConvNext` backbone  for `ADE20K`
  -  Adversarial-training: config-file: [ade20k_convnext.yaml](/configs/ade20k_convnext.yaml) set `BACKBONE` in `MODEL` to `CONVNEXT-S_CVST` and `CONVNEXT-T_CVST` for Small and Tiny models respectively. 
 
- For `UperNet` with `ConvNext` backbone for `PASCALVOC`
  	-  Adversarial-training: config-file: [pascalvoc_convnext.yaml](/configs/pascalvoc_convnext.yaml) set `BACKBONE` in `MODEL` to `CONVNEXT-S_CVST` and `CONVNEXT-T_CVST` for Small and Tiny models respectively. 
   
- For `SegMenter` with `Vit-S` backbone for `ADE20K` dataset
  
	-  Adversarial-training: config-file: [ade20k_segmenter.yaml](/configs/ade20k_segmenter.yaml) 

For clean-training: set `ADVERSARIAL` to FALSE in respective config-file.

_________________________________

<h4>Citation</h4>

If you use our code/models consider citing us with the follwong BibTex entry:
```
@@inproceedings{croce2023robust,
 title={Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models}, 
 author={Francesco Croce and Naman D Singh and Matthias Hein},
 year={2024},
 journal={ECCV}}
```
<h5>Acknowledgements</h5>

The code in this repository is partially based on the following publically available codebases.

1. [https://github.com/hszhao/semseg](https://github.com/hszhao/semseg)
2. [https://github.com/rstrudel/segmenter](https://github.com/rstrudel/segmenter) 
3. [https://github.com/facebookresearch/ConvNeXt/tree/main/semantic_segmentation](https://github.com/facebookresearch/ConvNeXt/tree/main/semantic_segmentation) 
4. [https://huggingface.co/docs/transformers/main/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation](https://huggingface.co/docs/transformers/main/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation)
