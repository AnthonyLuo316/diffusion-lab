"""
tests/test_shapes.py
Shape-correctness smoke tests for all diffusion_lab modules.

Run with:  pytest tests/ -v

Each test constructs a minimal model (tiny batch, small UNet/MLP) and
verifies that forward passes, loss calls, and sampler outputs have the
expected tensor shapes.  These tests do NOT require a GPU or trained
checkpoints — they use random weights and run in seconds on CPU.
"""

import math
import pytest
import torch

# ── Mark all tests as using CPU only ──────────────────────────────────────
DEVICE = "cpu"
B  = 2           # batch size
C  = 1           # image channels (MNIST)
H  = 28          # spatial height
W  = 28          # spatial width
D  = 2           # toy 2-D dimensionality
T  = 20          # short diffusion chain for speed


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def vp_schedule():
    from diffusion_lab.schedulers.schedules import VPSchedule
    return VPSchedule(T=T, schedule="cosine")


@pytest.fixture(scope="module")
def small_unet():
    from diffusion_lab.nn.unet import SmallUNet
    return SmallUNet(
        in_channels=C, out_channels=C,
        base_channels=8, time_embed_dim=16, dropout=0.0,
    ).eval()


@pytest.fixture(scope="module")
def small_unet_cond():
    from diffusion_lab.nn.unet import SmallUNet
    return SmallUNet(
        in_channels=C, out_channels=C,
        base_channels=8, time_embed_dim=16, dropout=0.0,
        num_classes=10,
    ).eval()


@pytest.fixture(scope="module")
def time_mlp():
    from diffusion_lab.nn.mlp import TimeMLP
    return TimeMLP(in_dim=D, hidden_dim=16, out_dim=D, n_layers=2).eval()


# ============================================================
# Scheduler
# ============================================================

class TestVPSchedule:
    def test_alpha_bar_shape(self, vp_schedule):
        assert vp_schedule.alpha_bar.shape == (T + 1,)

    def test_alpha_bar_monotone(self, vp_schedule):
        ab = vp_schedule.alpha_bar[1:]
        diffs = ab[1:] - ab[:-1]
        assert (diffs <= 0).all(), "alpha_bar should be non-increasing"

    def test_alpha_bar_endpoints(self, vp_schedule):
        assert vp_schedule.alpha_bar[0].item() == pytest.approx(1.0, abs=1e-4)
        assert vp_schedule.alpha_bar[T].item() < 0.05

    def test_broadcast(self, vp_schedule):
        t = torch.tensor([1, 5, T], dtype=torch.long)
        ab = vp_schedule.broadcast("alpha_bar", t, ndim=4)
        assert ab.shape == (3, 1, 1, 1)

    def test_cosine_vs_linear_different(self):
        from diffusion_lab.schedulers.schedules import VPSchedule
        cos = VPSchedule(T=T, schedule="cosine")
        lin = VPSchedule(T=T, schedule="linear")
        assert not torch.allclose(cos.alpha_bar, lin.alpha_bar)


# ============================================================
# SmallUNet
# ============================================================

class TestSmallUNet:
    def test_forward_shape(self, small_unet):
        x = torch.randn(B, C, H, W)
        t = torch.rand(B)
        out = small_unet(x, t)
        assert out.shape == (B, C, H, W)

    def test_forward_with_class_cond(self, small_unet_cond):
        x = torch.randn(B, C, H, W)
        t = torch.rand(B)
        c = torch.randint(0, 11, (B,))
        out = small_unet_cond(x, t, c)
        assert out.shape == (B, C, H, W)

    def test_null_token_zero(self, small_unet_cond):
        """Class index 0 (null token) should not crash and produce valid output."""
        x = torch.randn(B, C, H, W)
        t = torch.rand(B)
        c = torch.zeros(B, dtype=torch.long)
        out = small_unet_cond(x, t, c)
        assert out.shape == (B, C, H, W)
        assert torch.isfinite(out).all()

    def test_no_nan_in_output(self, small_unet):
        x = torch.randn(B, C, H, W)
        t = torch.rand(B)
        out = small_unet(x, t)
        assert torch.isfinite(out).all()

    def test_optional_c_none(self, small_unet):
        """Unconditional UNet should accept c=None gracefully."""
        x = torch.randn(B, C, H, W)
        t = torch.rand(B)
        out = small_unet(x, t, c=None)
        assert out.shape == (B, C, H, W)


# ============================================================
# DDPM
# ============================================================

class TestDDPM:
    @pytest.fixture(scope="class")
    def ddpm(self, vp_schedule, small_unet):
        from diffusion_lab.models.ddpm import DDPM
        return DDPM(network=small_unet, schedule=vp_schedule,
                    prediction="epsilon", loss_weight="uniform")

    def test_q_sample_shape(self, ddpm):
        x0 = torch.randn(B, C, H, W)
        t  = torch.randint(1, T + 1, (B,))
        xt = ddpm.q_sample(x0, t, torch.randn_like(x0))
        assert xt.shape == (B, C, H, W)

    def test_loss_scalar(self, ddpm):
        x0   = torch.randn(B, C, H, W)
        loss = ddpm.loss(x0)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_p_sample_shape(self, ddpm):
        x = torch.randn(B, C, H, W)
        t = torch.tensor([5] * B)
        out = ddpm.p_sample(x, t)
        assert out.shape == (B, C, H, W)

    @pytest.mark.parametrize("prediction", ["epsilon", "x0", "v"])
    def test_prediction_modes(self, vp_schedule, small_unet, prediction):
        from diffusion_lab.models.ddpm import DDPM
        model = DDPM(network=small_unet, schedule=vp_schedule, prediction=prediction)
        x0    = torch.randn(B, C, H, W)
        loss  = model.loss(x0)
        assert torch.isfinite(loss)


# ============================================================
# DDIM Sampler
# ============================================================

class TestDDIMSampler:
    @pytest.fixture(scope="class")
    def ddim_sampler(self, vp_schedule, small_unet):
        from diffusion_lab.models.ddpm import DDPM
        from diffusion_lab.models.ddim import DDIMSampler
        ddpm = DDPM(network=small_unet, schedule=vp_schedule)
        return DDIMSampler(ddpm)

    def test_sample_shape(self, ddim_sampler):
        samples = ddim_sampler.sample(
            shape=(B, C, H, W), device=DEVICE, num_steps=5, eta=0.0
        )
        assert samples.shape == (B, C, H, W)

    def test_stochastic_sample_shape(self, ddim_sampler):
        samples = ddim_sampler.sample(
            shape=(B, C, H, W), device=DEVICE, num_steps=5, eta=1.0
        )
        assert samples.shape == (B, C, H, W)

    def test_deterministic_reproducibility(self, ddim_sampler):
        """η=0 DDIM is deterministic given the same x_T."""
        xT = torch.randn(B, C, H, W)
        s1 = ddim_sampler.sample(
            (B, C, H, W), device=DEVICE, num_steps=5, eta=0.0, x_T=xT.clone()
        )
        s2 = ddim_sampler.sample(
            (B, C, H, W), device=DEVICE, num_steps=5, eta=0.0, x_T=xT.clone()
        )
        assert torch.allclose(s1, s2)


# ============================================================
# CFG (CondDDPM + CFGSampler)
# ============================================================

class TestCFG:
    @pytest.fixture(scope="class")
    def cond_ddpm(self, vp_schedule, small_unet_cond):
        from diffusion_lab.models.guidance import CondDDPM
        return CondDDPM(network=small_unet_cond, schedule=vp_schedule,
                        p_uncond=0.1, prediction="epsilon")

    def test_loss_shape(self, cond_ddpm):
        x0     = torch.randn(B, C, H, W)
        labels = torch.randint(1, 11, (B,))
        loss   = cond_ddpm.loss(x0, labels)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_net_cfg_shape(self, cond_ddpm):
        xt     = torch.randn(B, C, H, W)
        t_norm = torch.rand(B)
        c      = torch.randint(1, 11, (B,))
        out    = cond_ddpm._net_cfg(xt, t_norm, c, w=3.0)
        assert out.shape == (B, C, H, W)

    def test_cfg_sampler_ddim_shape(self, cond_ddpm):
        from diffusion_lab.models.guidance import CFGSampler
        sampler = CFGSampler(cond_ddpm, guidance_scale=2.0)
        labels  = torch.randint(1, 11, (B,))
        samples = sampler.sample_ddim(
            (B, C, H, W), labels, device=DEVICE, num_steps=3, eta=0.0
        )
        assert samples.shape == (B, C, H, W)

    def test_cfg_w0_equals_unconditional(self, cond_ddpm):
        """w=0 should ignore class labels — two calls with different labels should
        give very different outputs when w>0 but the same score when w=0."""
        xt     = torch.randn(1, C, H, W)
        t_norm = torch.tensor([0.5])
        c1     = torch.tensor([1])
        c2     = torch.tensor([9])
        # w=0: output should be identical regardless of c
        out0_c1 = cond_ddpm._net_cfg(xt, t_norm, c1, w=0.0)
        out0_c2 = cond_ddpm._net_cfg(xt, t_norm, c2, w=0.0)
        assert torch.allclose(out0_c1, out0_c2), "w=0 should ignore class label"


# ============================================================
# EDM
# ============================================================

class TestEDM:
    @pytest.fixture(scope="class")
    def edm_model(self, small_unet):
        from diffusion_lab.models.edm import EDMPrecon
        return EDMPrecon(small_unet, sigma_data=0.5)

    def test_forward_shape(self, edm_model):
        x     = torch.randn(B, C, H, W)
        sigma = torch.rand(B) * 5 + 0.01
        out   = edm_model(x, sigma)
        assert out.shape == (B, C, H, W)

    def test_loss_scalar(self, edm_model):
        x0   = torch.randn(B, C, H, W)
        loss = edm_model.loss(x0)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_precon_scalars_shapes(self, edm_model):
        sigma = torch.logspace(-2, 2, 10)
        for fn_name in ("c_skip", "c_out", "c_in", "c_noise", "loss_weight"):
            out = getattr(edm_model, fn_name)(sigma)
            assert out.shape == sigma.shape, f"{fn_name} shape mismatch"

    def test_c_skip_plus_c_out_x_unit_var(self, edm_model):
        """
        At σ = σ_data, c_skip = c_out = 0.5 and the output equals
        0.5 * x + 0.5 * F_θ(…), which is the design target.
        """
        sd  = edm_model.sigma_data
        sig = torch.tensor([sd])
        cs  = edm_model.c_skip(sig).item()
        co  = edm_model.c_out(sig).item()
        assert cs == pytest.approx(0.5, abs=1e-5)
        assert co == pytest.approx(sd / math.sqrt(2), abs=1e-5)

    def test_heun_sampler_shape(self, edm_model):
        from diffusion_lab.models.edm import EDMSampler
        sampler = EDMSampler(edm_model, sigma_min=0.1, sigma_max=5.0)
        out = sampler.sample((B, C, H, W), num_steps=3, device=DEVICE)
        assert out.shape == (B, C, H, W)

    def test_edm_sigma_schedule_shape(self):
        from diffusion_lab.models.edm import edm_sigma_schedule
        sigs = edm_sigma_schedule(50, sigma_min=0.002, sigma_max=80.0)
        assert sigs.shape == (51,)          # N+1 (appended zero at end)
        assert sigs[-1].item() == pytest.approx(0.0)
        # Monotonically decreasing
        assert (sigs[:-1] >= sigs[1:]).all()


# ============================================================
# Inverse operators
# ============================================================

class TestInverseOperators:
    def _img(self):
        return torch.randn(B, C, H, W)

    def test_random_mask_shape(self):
        from diffusion_lab.models.inverse import RandomMaskOperator
        op  = RandomMaskOperator(keep_prob=0.5)
        out = op(self._img())
        assert out.shape == (B, C, H, W)

    def test_random_mask_zeros(self):
        from diffusion_lab.models.inverse import RandomMaskOperator
        op  = RandomMaskOperator(keep_prob=0.0)
        out = op(self._img())
        assert out.abs().max().item() == pytest.approx(0.0)

    def test_box_mask_shape(self):
        from diffusion_lab.models.inverse import BoxMaskOperator
        op  = BoxMaskOperator(box_size=8)
        out = op(self._img())
        assert out.shape == (B, C, H, W)

    def test_box_mask_centre_zeroed(self):
        from diffusion_lab.models.inverse import BoxMaskOperator
        op  = BoxMaskOperator(box_size=8)
        img = torch.ones(1, C, H, W)
        out = op(img)
        r0  = (H - 8) // 2
        c0  = (W - 8) // 2
        assert out[0, 0, r0:r0+8, c0:c0+8].abs().max().item() == pytest.approx(0.0)

    def test_blur_shape(self):
        from diffusion_lab.models.inverse import GaussianBlurOperator
        op  = GaussianBlurOperator(kernel_size=5, sigma=1.5, channels=C)
        out = op(self._img())
        assert out.shape == (B, C, H, W)

    def test_sr_downsample_shape(self):
        from diffusion_lab.models.inverse import SuperResolutionOperator
        op  = SuperResolutionOperator(scale=4)
        out = op(self._img())
        assert out.shape == (B, C, H // 4, W // 4)

    def test_sr_pinv_shape(self):
        from diffusion_lab.models.inverse import SuperResolutionOperator
        op  = SuperResolutionOperator(scale=4)
        y   = op(self._img())
        x_  = op.A_pinv(y, (B, C, H, W))
        assert x_.shape == (B, C, H, W)


# ============================================================
# Configs
# ============================================================

class TestConfigs:
    def test_ddpm_config_defaults(self):
        from diffusion_lab.configs import DDPMConfig
        cfg = DDPMConfig()
        assert cfg.T == 1000
        assert cfg.schedule == "cosine"

    def test_unet_config_as_kwargs(self):
        from diffusion_lab.configs import UNetConfig
        cfg = UNetConfig(base_channels=8, num_classes=5)
        kw  = cfg.as_kwargs()
        assert kw["base_channels"] == 8
        assert kw["num_classes"] == 5

    def test_dps_config_build_operator(self):
        from diffusion_lab.configs import DPSConfig
        from diffusion_lab.models.inverse import RandomMaskOperator
        cfg = DPSConfig(operator="random_mask", keep_prob=0.7)
        op  = cfg.build_operator()
        assert isinstance(op, RandomMaskOperator)
        assert op.keep_prob == pytest.approx(0.7)

    def test_override_field(self):
        from diffusion_lab.configs import EDMConfig
        cfg = EDMConfig(sigma_max=50.0, num_steps=20)
        assert cfg.sigma_max == pytest.approx(50.0)
        assert cfg.num_steps == 20
