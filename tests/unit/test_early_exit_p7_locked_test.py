import io
import tarfile
from collections import Counter

import numpy as np
from scipy.io import savemat

from scripts.analysis import evaluate_early_exit_p7_locked_test as evaluator


def add_member(tar, name, content):
    info = tarfile.TarInfo(name)
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))


def test_official_subset_uses_devkit_ids_and_extracts_only_frozen_classes(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, 'ROOT', tmp_path)
    monkeypatch.setattr(evaluator, 'OUT', tmp_path / 'out')
    evaluator.OUT.mkdir()
    (tmp_path / 'data/imagenet100').mkdir(parents=True)
    classes = [f'n{i:08d}' for i in range(1, 101)]
    meta = np.empty((101, 5), dtype=object)
    for i in range(101):
        meta[i] = [i + 1, f'n{i+1:08d}', 'class', 'description', 0]
    mat = io.BytesIO()
    savemat(mat, {'synsets': meta})
    devkit = tmp_path / 'devkit.tar.gz'
    labels = [i for i in range(1,101) for _ in range(50)] + [101] * 45000
    with tarfile.open(devkit, 'w:gz') as tar:
        add_member(tar, 'ILSVRC2012_devkit_t12/data/meta.mat', mat.getvalue())
        add_member(tar, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt', '\n'.join(map(str, labels)).encode())
    archive = tmp_path / 'img_val.tar'
    with tarfile.open(archive, 'w') as tar:
        for i in range(1,5002):
            add_member(tar, f'ILSVRC2012_val_{i:08d}.JPEG', b'synthetic-image')
    lock = {'classes': classes, 'archives': {str(p): evaluator.sha256(p) for p in (devkit, archive)}}
    destination = evaluator.prepare(lock)
    assert Counter(p.parent.name for p in destination.rglob('*.JPEG')) == Counter({c:50 for c in classes})
    assert not (destination / 'n00000101').exists()
