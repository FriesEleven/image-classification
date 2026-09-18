# Minimal core evidence — completed

All scientific negative results are retained. Technical completion does not imply constraint success.

See shared_vs_individual_summary.csv and a3_joint_summary.csv; policy selection is post-hoc development analysis. I uses target fitting information.

| Dataset | Full-test policy accuracy (%) | Full-test MAC saving (%) | Resident subset latency saving (%) |
|---|---:|---:|---:|
| cifar10 | 87.0833 | 32.9436 | 19.1002 ± 2.3661 |
| cifar100 | 57.6367 | 15.1618 | -1.4961 ± 1.4504 |

Full test =10000 images/model; timed workload =1000 unique same-test images, repeated five rounds. These scopes are distinct.
Official datasets were used only after explicit authorization for six fixed A3 checkpoints and two locked thresholds. No threshold search on official data.
All graph/table claims must retain earlier large-batch slowdowns and P6/P7 stop/failure results.
