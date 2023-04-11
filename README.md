<h3>Some code for semantic segmentation.</h3>

- Currently tested for UperNet with ConvNext backbone for ADE20K dataset.
- UperNet code taken from Huggingface transformers UperNetforSegmentation.
- ConvNext code taken from the official repo.
- Aux_head loss seems to be very important in training these models.
- IGNORE_Label = -1.
- OPTIMIZER still does not have layerwise-decay scheduler like in original ConvNext.
- update config file in `configs` as required. Currently only useing `ConvNext-T`
- `run_train.sh` runs on A100 vms, and `runner.sh` on SLURM.
