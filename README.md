<h3>Some code for semantic segmentation.</h3>

- Tested for `UperNet` with `ConvNext` backbone and `PSPNet-RN50` backbone  for `ADE20K` dataset and `PASCALVOC`.
- UperNet code taken from Huggingface transformers UperNetforSegmentation.
- ConvNext code taken from the official repo.
- Aux_head loss seems to be very important in training these models.
- IGNORE_Label = -1. Currently not ignoring any pixel either for train or eval.
- OPTIMIZER still does not have layerwise-decay scheduler like in original ConvNext.
- update config file in `configs` as required. Currently only useing `ConvNext-T`
- `run_train.sh` runs on A100 vms, and `runner.sh` on SLURM.
- `train.py` has the main training script.
- `infer.py` does evaluation on standard 1.4k images of PASCALVOC `adversarial` flag makes evaluation of a PGD or just the clean model.
- `seg_map.py` creates images of the image-GT-output-map. 