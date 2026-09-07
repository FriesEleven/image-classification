# P7 ImageNet-100 source-gate result

The six serial source runs completed without official ImageNet validation access. A3 improved final-head model-selection accuracy over A0 by 1.41, 0.80, and 0.84 percentage points for seeds 77, 78, and 79.

The frozen source-calibration search selected one shared exit-8 confidence threshold of 0.850. Every source seed met the predeclared overall, balanced, worst-class, routing-coverage, and corrected-MAC constraints. The minimum source-seed MAC saving was 15.54%, with early-exit fractions from 27.64% to 27.99%.

The source gate therefore permits target training on the untouched seeds 80-82. Threshold selection or recalibration on those target seeds remains forbidden, and the official ImageNet validation subset remains unprepared and inaccessible.
