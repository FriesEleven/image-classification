# P7 ImageNet-100 design freeze

P7 replaces the planned Tiny ImageNet-200 download with a fixed ImageNet-100
archive already present in the server's public dataset mount.  This avoids a
download, uses 126,689 ILSVRC2012 training images at 224x224 resolution, and
freezes the exact 100 synsets and source archive hash.  It must be described as
ImageNet-100, not ImageNet-1K.

The label-free architecture profile selected MobileNetV2 feature 8 as the
deployable exit (43.77% of final Conv/Linear MACs) and feature 15 as the
training-only auxiliary exit (80.42%).  The simplified A3 recipe from P5-C is
used without knowledge distillation.

Source training is six serial runs: A0/A3 for seeds 77--79.  Only if both the
final-head and shared-policy gates pass will target seeds 80--82 be prepared.
The official ImageNet validation subset is not prepared and remains behind a
separate one-time gate.
