import unittest

import torch

from yolox.hapq.cost_model import HAPQBudget, HAPQCostModel, LayerCostSpec
from yolox.hapq.problem import HAPQSearchSpace
from yolox.hapq.quantization import quantize_tensor_symmetric


class TestHAPQCore(unittest.TestCase):
    def test_quantization_range(self):
        x = torch.tensor([-10.0, -1.25, 0.0, 1.25, 10.0])
        q = quantize_tensor_symmetric(x, bit_width=8)
        self.assertEqual(q.shape, x.shape)
        self.assertTrue(torch.all(torch.isfinite(q)))

    def test_cost_model_objective(self):
        budget = HAPQBudget(tau_lat=1000, tau_eng=1000, tau_dsp=1000, tau_bram=1000)
        model = HAPQCostModel(budget=budget)
        specs = [
            LayerCostSpec(
                name="l1",
                p_req=128,
                dense_synops=10000,
                activity=0.2,
                mask_keep_ratio=0.75,
                b_w=8,
                b_u=8,
                state_neurons=256,
                timesteps=4,
            )
        ]
        out = model.objective(det_loss=0.3, layer_specs=specs)
        self.assertGreater(out.total_loss, 0.0)
        self.assertGreater(out.resources.dsp, 0.0)
        self.assertGreater(out.resources.bram, 0.0)

    def test_search_space_sampling(self):
        space = HAPQSearchSpace(
            layer_names=["a", "b"],
            channel_choices=[64, 128],
            kernel_choices=[1, 3],
            depth_choices=[1, 2],
            bit_choices_w=[4, 8],
            bit_choices_u=[6, 8],
            block_size=8,
            leak_shift_choices=[2, 3],
        )
        cand = space.sample_candidate(rng=__import__("random").Random(0))
        self.assertEqual(len(cand.layers), 2)
        cand.validate()


if __name__ == "__main__":
    unittest.main()
