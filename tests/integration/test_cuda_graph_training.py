import copy

import pytest
import torch
from torch.amp import GradScaler

from image_classification.config import ExperimentConfig
from image_classification.models import build_model
from image_classification.training.cuda_graph import prepare_training_graph
from image_classification.training.engine import _train_epoch
from image_classification.training.evaluate import validate

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph integration")


class Counter:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


@pytest.mark.parametrize("model_type", ["hybrid_leaky", "csgha_v4"])
def test_graph_matches_eager_no_cache_and_preserves_partial_batch_eval_and_rng(model_type):
    torch.manual_seed(11)
    config = ExperimentConfig(model_type=model_type, se_positions=(1, 2), cbam_positions=(7, 8),
                              batch_size=4, cuda_graph=True, accumulation_steps=1,
                              num_workers=0, evaluate_test=False)
    eager = build_model(config).cuda().train()
    graph = copy.deepcopy(eager)
    state = {key: value.clone() for key, value in graph.state_dict().items()}
    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    prepare_training_graph(graph, 4, torch.device("cuda"), amp=True)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)
    assert all(torch.equal(value, graph.state_dict()[key]) for key, value in state.items())
    assert graph.state_dict().keys() == eager.state_dict().keys()
    batches = [(torch.randn(size, 3, 32, 32, device="cuda"), torch.arange(size, device="cuda") % 10)
               for size in (4, 2)]
    results = []
    for model in (eager, graph):
        torch.manual_seed(888)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = GradScaler("cuda", init_scale=8)
        scheduler = Counter()
        for _ in range(3):
            metrics = _train_epoch(model, batches, torch.nn.CrossEntropyLoss(), optimizer, scheduler,
                                   scaler, config, torch.device("cuda"))
        results.append((metrics, scheduler.steps, torch.cuda.get_rng_state().clone(), scaler.get_scale()))
    assert results[0][0] == results[1][0]
    # Tiny random batches can overflow AMP; both backends must skip identical
    # updates rather than forcing a scheduler step when the optimizer skipped.
    assert 0 < results[0][1] == results[1][1] <= 6
    assert results[0][3] == results[1][3]
    assert torch.equal(results[0][2], results[1][2])
    for key, value in eager.state_dict().items():
        # Graph backward can reorder the two guidance-gradient contributions;
        # only the shared scalar gate has a tiny float32 rounding difference.
        if key.endswith("guidance_scale"):
            torch.testing.assert_close(value, graph.state_dict()[key], rtol=0, atol=1e-7)
        else:
            assert torch.equal(value, graph.state_dict()[key]), key
    assert graph._training_graph_replays == 3
    assert graph._training_graph_fallbacks == 3
    assert all(int(value) == 6 for key, value in graph.state_dict().items() if key.endswith("num_batches_tracked"))
    eager_eval = validate(eager, batches, torch.nn.CrossEntropyLoss(), torch.device("cuda"))
    graph_eval = validate(graph, batches, torch.nn.CrossEntropyLoss(), torch.device("cuda"))
    assert eager_eval[0] == graph_eval[0]
    assert (eager_eval[3] == graph_eval[3]).all()
    assert graph._training_graph_replays == 3
