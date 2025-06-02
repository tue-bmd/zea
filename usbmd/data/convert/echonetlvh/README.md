## EchoNet-LVH conversion
- For EchoNet-LVH conversion we first identify the scan cones in each video, since these
  vary from file to file. This is done by the `precompute_crop.py` script, which will
  produce as output a .csv file `cone_parameters.csv`. The cone parameters will indicate
  how the video should be cropped in order to bound the cone and remove margins.
- Next, `convert_raw_to_usbmd.py` can be run to convert the dataset to usbmd format,
  with cropping and scan conversion. The measurement locations stored in `MeasurementsList.csv`
  are also updated to match the new cropping / padding coordinates.
- You can save the video and measurement plots for a converted video using `examples/echonetlvh/plot_sample.py`.