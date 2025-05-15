## EchoNet-LVH conversion
- For EchoNet-LVH conversion we first identify the scan cones in each video, since these
  vary from file to file. This is done by the `precompute_crop.py` script, which will
  produce as output a .csv 