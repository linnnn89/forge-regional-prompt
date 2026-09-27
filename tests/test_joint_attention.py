"""Run with Forge Python; accepts the Forge root as the only argument."""
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), sys.argv.pop(1)]
from backend.nn.anima import SelfCrossAttention
from lib_couple.anima import AttentionCoupleAnima
from lib_couple.joint_attention import JointAttentionAnima, spatial_bias


def cpu_attention(q, k, v, transformer_options=None):
    q, k, v = [t.reshape(t.shape[0], -1, t.shape[-2], t.shape[-1]) for t in (q, k, v)]
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2)).transpose(1, 2).flatten(2)


class JointAttentionTests(unittest.TestCase):
    def tearDown(self):
        AttentionCoupleAnima.unpatch()

    def fixture(self):
        torch.manual_seed(43)
        cross = SelfCrossAttention(16, 16, n_heads=2, head_dim=8)
        self_attn = SelfCrossAttention(16, n_heads=2, head_dim=8)
        dit = torch.nn.ModuleList([cross, self_attn])
        dit.patch_spatial = 2
        model = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dit))
        left = torch.ones(1, 16, 48)
        left[..., 32:] = 0
        args = {}
        for i, (mask, length) in enumerate(zip([torch.ones_like(left), left, 1-left], [2, 3, 4]), 1):
            args[f'mask_{i}'] = mask
            args[f'cond_{i}'] = [torch.randn(1, length, 16)]
        return model, cross, self_attn, args

    def test_continuous_prior_and_length_neutrality(self):
        planes = torch.tensor([[1., 1., 1.], [1., .5, 0.], [0., .5, 1.]])
        bias = spatial_bias(planes, [1, 2, 3], math.log(4))
        # QK=0: each text slot has total prior 1, 1/.625/.25, .25/.625/1.
        values = torch.tensor([10., 2., 2., 8., 8., 8.])
        result = (bias.softmax(-1) * values).sum(-1).flatten()
        expected = torch.tensor([14/2.25, 16.25/2.25, 18.5/2.25])
        torch.testing.assert_close(result, expected)
        self.assertTrue(torch.isfinite(bias).all())
        duplicated = spatial_bias(planes, [2, 4, 6], math.log(4))
        torch.testing.assert_close((duplicated.softmax(-1)*values.repeat_interleave(2)).sum(-1).flatten(), expected)
        no_position = spatial_bias(planes, [1, 2, 3], 0)
        torch.testing.assert_close((no_position.softmax(-1)*values).sum(-1), torch.full((1, 1, 3), 20/3))

    def test_real_anima_forward_batch_negative_self_and_cleanup(self):
        model, cross, self_attn, args = self.fixture()
        original = SelfCrossAttention.forward
        with patch.object(SelfCrossAttention, 'torch_attention_op', staticmethod(cpu_attention)):
            for batch, flags in [(1, [0, 1]), (2, [1, 0]), (2, [0])]:
                AttentionCoupleAnima.unpatch()
                x = torch.randn(batch*len(flags), 3, 16)
                context = torch.randn(batch*len(flags), 1, 2, 16)
                opts = {'cond_or_uncond': flags}
                self_before = original(self_attn, x, None, None, opts)
                _, stats = JointAttentionAnima.patch_dit(model, 48, 16, args, math.log(4))
                actual = cross(x, context, None, opts)
                torch.testing.assert_close(self_attn(x, None, None, opts), self_before, rtol=0, atol=0)
                priors = torch.tensor([[1., 1., .25], [1., 1., .25], [1., .25, 1.]])
                prior_tokens = torch.cat([priors[:, i:i+1].expand(3, n)/n for i, n in enumerate([2, 3, 4])], 1)
                text = torch.cat([args[f'cond_{i}'][0] for i in range(1, 4)], 1).expand(batch, -1, -1)
                for i, flag in enumerate(flags):
                    part = slice(i*batch, (i+1)*batch)
                    if flag:
                        expected_negative = original(cross, x[part], context[part], None, opts)
                        torch.testing.assert_close(actual[part], expected_negative, rtol=0, atol=0)
                    else:
                        q, k, v = cross.compute_qkv(x[part], text)
                        q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
                        probabilities = (q@k.transpose(-1, -2)/math.sqrt(8)+prior_tokens.log()).softmax(-1)
                        expected = cross.output_proj((probabilities@v).transpose(1, 2).flatten(2))
                        torch.testing.assert_close(actual[part], expected)
                self.assertGreater(stats['joint_calls'], 0)
            AttentionCoupleAnima.unpatch()
            self.assertIs(SelfCrossAttention.forward, original)

    def test_disabled_layers_and_invalid_weights(self):
        model, _, _, args = self.fixture()
        args['mask_3'].zero_()
        _, stats = JointAttentionAnima.patch_dit(model, 48, 16, args)
        self.assertEqual(stats['conditions'], 2)
        args['mask_2'][..., 0] = -1
        with self.assertRaisesRegex(ValueError, 'invalid mask'):
            JointAttentionAnima.patch_dit(model, 48, 16, args)
        self.assertFalse(getattr(SelfCrossAttention.forward, '_couple', False))


if __name__ == '__main__':
    unittest.main()
