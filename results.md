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
