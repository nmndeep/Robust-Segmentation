<h3>Some results for UperNet ConvNeXt</h3>

- lr = 1e-4, wd = 5e-2, img_size=512, with aux_loss coeff. = 0.4

for ADE20k dataset
- IM-1k pretrained model has around 42.9% mIou, the original paper has 46%.
- IM-1k pretrained CVST  -- running.

- Currently tested for UperNet with ConvNext backbone for ADE20K dataset.
- Aux_head loss seems to be very important in training these models.
- IGNORE_Label = -1.
- OPTIMIZER still does not have layerwise-decay scheduler like in original ConvNext.


  | init_method  |val mIou% |
 -|--------------|----------|
1 | ConvNexT-T   |   43.9%	| 46.0
2 | ConvNexT-CvsT|   42.5%	|
3 | CvNxT-CvStRob|   41.4%	|




PASCALVOC-aug

UperNetForSemanticSegmentation - ConvNeXt-T_CVST      
Clean mIoU 88.02
PGD: eps: 2/255, iter : 5 -- mIoU 51.8
PGD: eps: 4/255, iter : 5 -- mIoU 29.55	
PGD: eps: 4/255, iter : 20 -- mIoU 16.3	
PGD: eps: 4/255, iter : 100 -- mIoU 15.37
PGD: eps: 8/255, iter : 100 -- mIoU 3.89


PSPNet - ResNet-50
Clean mIoU 86.7
PGD: eps: 2/255, iter : 5 -- mIoU 46.9
PGD: eps: 4/255, iter : 5 -- mIoU 28.55	
PGD: eps: 4/255, iter : 100 -- mIoU 13.98
PGD: eps: 8/255, iter : 100 -- mIoU 4.91


UperNetForSemanticSegmentation - ConvNeXt-T_CVST_ROB
Clean mIoU 86.85	
PGD: eps: 2/255. iter : 5 -- mIoU 67.97	
PGD: eps: 4/255, iter : 5 -- mIoU 49.23	
PGD: eps: 4/255, iter : 20 -- mIoU 32.41
PGD: eps: 4/255, iter : 100 -- mIoU 29.43
PGD: eps: 8/255, iter : 100 -- mIoU 7.71



