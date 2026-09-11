"""C24b / U3.4b3: exact deterministic keyframe decoder bridge on Comfy modules.

This supersedes C24's hand-reconstructed block probe.

Algorithmic behavior is fixed by the frozen LTX sources:

* ``NABlock.forward_with_keyframes``:
  shared norm1 -> dual-stream attention -> independent residual adds ->
  shared norm2/SwiGLU residual on both streams. Invalid keyframe planes are NOT
  re-zeroed inside a block.
* ``NeighborhoodAttention3D.forward_with_keyframes``:
  video uses deterministic absolute RoPE at local integer times; keyframes use
  the same Q/K/V weights, Q/K norm, query scale and absolute RoPE at the exact
  fractional keyframe stage times; both then enter the validated U3.4b2 joint
  attention softmax and the same output projection.
* ``_run_det_stage_with_keyframes``:
  execute every dual-stream block, normal upsample for video, official isolated
  plane upsample for keyframes, rebuild next-stage times from global pixel
  indices, then mask invalid keyframe planes at the STAGE boundary.

The heavy weights remain the loaded Comfy ``NADiffusionDecoder`` modules.  The
bridge only supplies semantics missing from current Comfy and uses Comfy's own
VAE ModelPatcher/device lifecycle and ``comfy_kitchen.rms_rope_`` primitive.

For memory parity with the official tiled decoder, the real full-volume
execution stops after deterministic stages 1-3, i.e. at the official
``forward_stages_1_to_3_with_keyframes`` output.  Stage 4 is also executed on a
small real feature/keyframe probe to validate its exact bridge, but a gigantic
full stage-4 context is deliberately NOT materialized: official keyframe decode
runs stage 4 together with stage 5 per tile.  C25 will consume the stage-4 input
state produced here and execute the real tile path.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from .decoder_geometry_cache import cached_geometry

from .decoder_det_stage_context import DFRDeterministicDecoderKeyframeContext
from .decoder_det_stage_official_preflight import (
    DFROfficialDeterministicStagePreflight,
    keyframe_clip_times_official,
    upsample_keyframe_planes_official,
)
from .decoder_joint_attention import DFRJointDecoderAttention, joint_na3d


EXACT_DET_PORT_VERSION = 2
EXACT_DET_EXECUTION_VERSION = 1


@dataclass(frozen=True)
class DFRExactDeterministicDualStreamPort:
    version: int
    source_contract: str
    runtime_backbone: str
    joint_attention_backend: str
    num_slots: int
    qkv_storage_mapping: str
    rope_execution_mapping: str
    mlp_execution_mapping: str
    invalid_planes_rezeroed_inside_block: bool
    invalid_planes_rezeroed_after_upsample: bool
    stages_1_to_3_full_volume: bool
    stage_4_tiled_with_stage_5: bool
    decoder_mutated: bool


@dataclass(frozen=True)
class DFRDeterministicStageExecution:
    version: int
    source_contract: str
    execution_mode: str
    device: str
    dtype: str
    input_video_shape: tuple[int, ...]
    ghost_pad_latent_frames: int
    padded_video_shape: tuple[int, ...]
    stage_video_shapes: tuple[tuple[int, ...], ...]
    stage_keyframe_shapes: tuple[tuple[int, ...], ...]
    stage_keyframe_times: tuple[tuple[float, ...], ...]
    stage_keyframe_valid_counts: tuple[int, ...]
    stage4_input_video: torch.Tensor
    stage4_input_keyframes: torch.Tensor
    stage4_input_times: torch.Tensor
    stage4_input_valid: torch.Tensor
    stage4_probe_video_input_shape: tuple[int, ...]
    stage4_probe_keyframe_input_shape: tuple[int, ...]
    stage4_probe_video_output_shape: tuple[int, ...]
    stage4_probe_keyframe_output_shape: tuple[int, ...]
    stage4_probe_output_times: tuple[float, ...]
    stage4_probe_cross_stream_influence: float
    stage4_probe_invalid_keyframe_zero_error: float
    decoder_parameters_unchanged: bool
    cpu_rng_unchanged: bool
    cuda_rng_unchanged: bool
    stage5_called: bool
    decoder_mutated: bool


@dataclass
class _KeyframeStream:
    x: torch.Tensor
    times: torch.Tensor
    valid: torch.Tensor

    def masked(self) -> "_KeyframeStream":
        mask = self.valid.to(dtype=self.x.dtype, device=self.x.device)[None, :, None, None, None]
        return _KeyframeStream(x=self.x * mask, times=self.times, valid=self.valid)


def prepare_exact_deterministic_dual_stream_port(
    official_preflight: DFROfficialDeterministicStagePreflight,
    joint_decoder_attention: DFRJointDecoderAttention,
) -> tuple[DFRExactDeterministicDualStreamPort, str]:
    if not isinstance(official_preflight, DFROfficialDeterministicStagePreflight):
        raise ValueError(
            f"Expected DFROfficialDeterministicStagePreflight, got {type(official_preflight).__name__}."
        )
    if not isinstance(joint_decoder_attention, DFRJointDecoderAttention):
        raise ValueError(
            f"Expected DFRJointDecoderAttention, got {type(joint_decoder_attention).__name__}."
        )
    if not official_preflight.architecture_matches_checkpoint_config:
        raise ValueError("C24b requires the validated C23b checkpoint/runtime mapping.")
    if not official_preflight.keyframe_plane_helper_exact:
        raise ValueError("C24b requires the validated official keyframe plane-upsample helper.")
    if joint_decoder_attention.source_backend not in {
        "official_fallback_joint_eager",
        "official_auto_joint_triton_cuda_eager_fallback",
    }:
        raise ValueError(
            "C24b requires the frozen official joint-attention semantics validated in U3.4b2."
        )

    state = DFRExactDeterministicDualStreamPort(
        version=EXACT_DET_PORT_VERSION,
        source_contract=(
            "frozen_NABlock.forward_with_keyframes+"
            "NeighborhoodAttention3D.forward_with_keyframes+"
            "det_qkv_rope_at_times+_run_det_stage_with_keyframes"
        ),
        runtime_backbone="comfy.NADiffusionDecoder_existing_weights",
        joint_attention_backend=joint_decoder_attention.source_backend,
        num_slots=int(joint_decoder_attention.num_slots),
        qkv_storage_mapping="comfy_fused_qkv_same_checkpoint_projection",
        rope_execution_mapping="comfy_kitchen.rms_rope_with_official_fractional_times",
        mlp_execution_mapping="comfy_existing_chunked_SwiGLU_same_weights_and_formula",
        invalid_planes_rezeroed_inside_block=False,
        invalid_planes_rezeroed_after_upsample=True,
        stages_1_to_3_full_volume=True,
        stage_4_tiled_with_stage_5=True,
        decoder_mutated=False,
    )
    report = (
        f"PASS=True; stage=U3.4b3_C24b_prepare_exact_det_port; version={state.version}; "
        f"source_contract={state.source_contract}; runtime_backbone={state.runtime_backbone}; "
        f"joint_attention_backend={state.joint_attention_backend}; num_slots={state.num_slots}; "
        f"qkv_storage_mapping={state.qkv_storage_mapping}; rope_execution_mapping={state.rope_execution_mapping}; "
        f"mlp_execution_mapping={state.mlp_execution_mapping}; "
        f"invalid_planes_rezeroed_inside_block={state.invalid_planes_rezeroed_inside_block}; "
        f"invalid_planes_rezeroed_after_upsample={state.invalid_planes_rezeroed_after_upsample}; "
        f"stages_1_to_3_full_volume={state.stages_1_to_3_full_volume}; "
        f"stage_4_tiled_with_stage_5={state.stage_4_tiled_with_stage_5}; decoder_mutated=False."
    )
    return state, report


def _require_comfy_runtime():
    try:
        import comfy.model_management as model_management
        import comfy.ldm.lightricks.vae.na_diffusion_decoder as comfy_na
        import comfy_kitchen
    except Exception as exc:  # pragma: no cover - exercised in Comfy runtime
        raise RuntimeError(
            "C24b exact deterministic execution requires the current Comfy LTX DiffVAE runtime."
        ) from exc
    return model_management, comfy_na, comfy_kitchen


@cached_geometry
def _rope_axis_part(comfy_na, dim, base, device, length, positions=None):
    inv = comfy_na.rope_inv_freqs(dim, base, device=device)
    pos = (torch.arange(length, dtype=torch.float32, device=device) if positions is None
           else positions.to(device=device, dtype=torch.float32))
    ang = pos[:, None] * inv[None, :]
    c, sine = ang.cos(), ang.sin()
    return torch.stack([c, -sine, sine, c], dim=-1).reshape(c.shape[0], 1, 1, c.shape[1], 2, 2)


def _rope_matrices_for_times(
    comfy_na: Any,
    attn: Any,
    temporal_positions: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
    *,
    height_positions: torch.Tensor | None = None,
    width_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Comfy rms_rope matrix layout, with official caller-supplied keyframe T positions.

    Current Comfy's integer-time helper is documented as matching ltx-core RoPE
    numerics.  This is exactly that construction, except the temporal arange is
    replaced by the official fractional ``keyframe_clip_times`` values.
    """
    rope_split = tuple(int(v) for v in attn.rope_split)
    t_pos = temporal_positions.to(device=device, dtype=torch.float32)
    if ((height_positions is not None and height_positions.numel() != height)
            or (width_positions is not None and width_positions.numel() != width)):
        raise ValueError("RoPE position lengths must match the tensor geometry.")
    parts = [
        _rope_axis_part(comfy_na, dim, float(attn.rope_base), device, length, positions)
        for dim, length, positions in zip(
            rope_split, (t_pos.numel(), height, width),
            (t_pos, height_positions, width_positions), strict=True,
        )
    ]

    t = int(t_pos.numel())
    freq = torch.cat(
        [
            parts[0].expand(t, height, width, -1, 2, 2),
            parts[1].transpose(0, 1).expand(t, height, width, -1, 2, 2),
            parts[2].movedim(0, 2).expand(t, height, width, -1, 2, 2),
        ],
        dim=3,
    )
    return freq.reshape(1, t * height * width, 1, -1, 2, 2)


def _det_qkv_rope_comfy_backbone(
    comfy_na: Any,
    comfy_kitchen: Any,
    attn: Any,
    x: torch.Tensor,
    temporal_positions: torch.Tensor,
    *,
    width_positions: torch.Tensor | None = None,
    separate_qkv: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official det-QKV semantics using Comfy's checkpoint representation/operators."""
    if x.ndim != 5:
        raise ValueError(f"Expected channels-last [B,T/P,H,W,C], got {tuple(x.shape)}.")
    batch, axis, height, width, _ = x.shape
    shape = (batch, axis, height, width, int(attn.num_heads), int(attn.head_dim))
    q = torch.empty(shape, dtype=x.dtype, device=x.device)
    k = torch.empty(shape, dtype=x.dtype, device=x.device)
    v = torch.empty(shape, dtype=x.dtype, device=x.device)

    # Preserve current Comfy's memory-bounded temporal QKV/RoPE execution.  This
    # changes only materialization strategy, not the frozen source formula.
    chunk = max(1, (2 ** 25) // max(int(height * width * attn.dim), 1))
    q_weight = (attn.q_norm.weight.detach() * float(attn.scale)).to(x.dtype)
    k_weight = attn.k_norm.weight.detach().to(x.dtype)
    positions = temporal_positions.to(device=x.device, dtype=torch.float32)
    for t0 in range(0, int(axis), int(chunk)):
        t1 = min(t0 + int(chunk), int(axis))
        cshape = (
            batch,
            t1 - t0,
            height,
            width,
            int(attn.num_heads),
            int(attn.head_dim),
        )
        if separate_qkv:
            # Official chunked attention deliberately uses three independent
            # projections so the merged 3*C activation is never resident.
            q_w, k_w, v_w = attn.qkv.weight.chunk(3, dim=0)
            if attn.qkv.bias is None:
                q_b = k_b = v_b = None
            else:
                q_b, k_b, v_b = attn.qkv.bias.chunk(3, dim=0)
            source = x[:, t0:t1]
            qc = torch.nn.functional.linear(source, q_w, q_b)
            q[:, t0:t1] = qc.reshape(cshape)
            del qc
            kc = torch.nn.functional.linear(source, k_w, k_b)
            k[:, t0:t1] = kc.reshape(cshape)
            del kc
            vc = torch.nn.functional.linear(source, v_w, v_b)
            v[:, t0:t1] = vc.reshape(cshape)
            del vc
        else:
            qc, kc, vc = attn.qkv(x[:, t0:t1]).chunk(3, dim=-1)
            q[:, t0:t1] = qc.reshape(cshape)
            k[:, t0:t1] = kc.reshape(cshape)
            v[:, t0:t1] = vc.reshape(cshape)
        freqs = _rope_matrices_for_times(
            comfy_na,
            attn,
            positions[t0:t1],
            int(height),
            int(width),
            x.device,
            width_positions=width_positions,
        )
        nt = int((t1 - t0) * height * width)
        for b in range(batch):
            comfy_kitchen.rms_rope_(
                q[b, t0:t1].reshape(1, nt, int(attn.num_heads), int(attn.head_dim)),
                k[b, t0:t1].reshape(1, nt, int(attn.num_heads), int(attn.head_dim)),
                freqs,
                q_weight,
                k_weight,
            )
    return q.contiguous(), k.contiguous(), v.contiguous()


def _forward_attention_with_keyframes_exact(
    comfy_na: Any,
    comfy_kitchen: Any,
    attn: Any,
    video_normed: torch.Tensor,
    keyframe_normed: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    *,
    num_slots: int,
    width_positions: torch.Tensor | None = None,
    separate_qkv: bool = False,
    timings: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = timings.phase if timings is not None else lambda name: nullcontext()
    with phase("stage5.attention.qkv_rope"):
        video_times = torch.arange(
            video_normed.shape[1], device=video_normed.device, dtype=torch.float32
        )
        q, k, v = _det_qkv_rope_comfy_backbone(
            comfy_na,
            comfy_kitchen,
            attn,
            video_normed,
            video_times,
            width_positions=width_positions,
            separate_qkv=separate_qkv,
        )
        kq, kk, kv = _det_qkv_rope_comfy_backbone(
            comfy_na,
            comfy_kitchen,
            attn,
            keyframe_normed,
            keyframe_times.to(device=keyframe_normed.device, dtype=torch.float32),
            width_positions=width_positions,
            separate_qkv=separate_qkv,
        )
    with phase("stage5.attention.joint_attention"):
        video_attn, keyframe_attn = joint_na3d(
            q,
            k,
            v,
            kq,
            kk,
            kv,
            keyframe_times.to(device=q.device, dtype=torch.float32),
            keyframe_valid.to(device=q.device, dtype=torch.bool),
            tuple(int(v) for v in attn.kernel_size),
            num_slots=int(num_slots),
        )
    dim = int(attn.dim)
    video_attn = video_attn.reshape(*video_normed.shape[:-1], dim)
    keyframe_attn = keyframe_attn.reshape(*keyframe_normed.shape[:-1], dim)

    # Same shared projection as official, executed with Comfy's temporal
    # chunking discipline so the bridge does not defeat native VAE memory bounds.
    def project_chunked(x: torch.Tensor) -> torch.Tensor:
        _b, axis, h, w, _c = x.shape
        chunk = max(1, (2 ** 25) // max(int(h * w * dim), 1))
        out = torch.empty_like(x)
        for t0 in range(0, int(axis), int(chunk)):
            t1 = min(t0 + int(chunk), int(axis))
            out[:, t0:t1] = attn.proj(x[:, t0:t1])
        return out

    with phase("stage5.attention.projection"):
        return project_chunked(video_attn), project_chunked(keyframe_attn)


def _forward_block_with_keyframes_exact(
    comfy_na: Any,
    comfy_kitchen: Any,
    block: Any,
    video_x: torch.Tensor,
    stream: _KeyframeStream,
    *,
    num_slots: int,
) -> tuple[torch.Tensor, _KeyframeStream]:
    """Direct port of frozen ``NABlock.forward_with_keyframes`` onto Comfy weights."""
    video_norm = block.norm1(video_x)
    keyframe_norm = block.norm1(stream.x)
    attn_out, keyframe_attn = _forward_attention_with_keyframes_exact(
        comfy_na,
        comfy_kitchen,
        block.attn,
        video_norm,
        keyframe_norm,
        stream.times,
        stream.valid,
        num_slots=num_slots,
    )
    video_x = video_x + attn_out
    keyframe_x = stream.x + keyframe_attn

    # Exact source formula is plain_mlp(x, mlp, norm2, tile).  Current Comfy's
    # chunked SwiGLU ``pre``/``add_to`` path is the same formula on the same
    # checkpoint weights without materializing the large pre-normalized tensor.
    video_x = block.mlp(video_x, pre=block.norm2, add_to=video_x)
    keyframe_x = block.mlp(keyframe_x, pre=block.norm2, add_to=keyframe_x)

    # IMPORTANT: official source deliberately does NOT mask here.
    return video_x, _KeyframeStream(x=keyframe_x, times=stream.times, valid=stream.valid)


def _run_det_stage_with_keyframes_exact(
    comfy_na: Any,
    comfy_kitchen: Any,
    decoder: Any,
    video_x: torch.Tensor,
    stream: _KeyframeStream,
    stage_i: int,
    pixel_frame_indices: torch.Tensor,
    *,
    drop_leading_frame: bool,
    next_time_origin: float,
    clip_start_frame: int,
    remaining_time_strides: tuple[int, ...],
    num_slots: int,
) -> tuple[torch.Tensor, _KeyframeStream]:
    video_x, stream = _run_det_stage_blocks_with_keyframes_exact(
        comfy_na,
        comfy_kitchen,
        decoder,
        video_x,
        stream,
        stage_i,
        num_slots=num_slots,
    )

    upsample = decoder.upsamples[int(stage_i)]
    video_x = upsample(video_x, drop_leading_frame=bool(drop_leading_frame))
    keyframe_x = upsample_keyframe_planes_official(upsample, stream.x)
    next_times = keyframe_clip_times_official(
        pixel_frame_indices,
        int(remaining_time_strides[int(stage_i) + 1]),
        int(clip_start_frame),
        extra_origin=float(next_time_origin),
    ).to(device=keyframe_x.device)
    stream = _KeyframeStream(
        x=keyframe_x,
        times=next_times,
        valid=stream.valid,
    ).masked()
    return video_x, stream


def _run_det_stage_blocks_with_keyframes_exact(
    comfy_na: Any,
    comfy_kitchen: Any,
    decoder: Any,
    video_x: torch.Tensor,
    stream: _KeyframeStream,
    stage_i: int,
    *,
    num_slots: int,
) -> tuple[torch.Tensor, _KeyframeStream]:
    """Run one deterministic stage without its trailing pixel-shuffle.

    This is the official deferred-Stage-4 boundary used by CHUNKED_EAGER.
    The existing full-stage helper above still owns all earlier stages and the
    combined-path oracle.
    """
    for block in decoder.det_stages[int(stage_i)]:
        video_x, stream = _forward_block_with_keyframes_exact(
            comfy_na,
            comfy_kitchen,
            block,
            video_x,
            stream,
            num_slots=num_slots,
        )
    return video_x, stream


def _sample_parameter_checksum(module: Any) -> tuple[tuple[str, float], ...]:
    rows: list[tuple[str, float]] = []
    for name, param in module.named_parameters():
        if param.numel() == 0:
            rows.append((name, 0.0))
            continue
        flat = param.detach().reshape(-1)
        count = min(4, int(flat.numel()))
        idx = torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()
        rows.append((name, float(flat.index_select(0, idx).float().sum().item())))
    return tuple(rows)


def _load_comfy_vae_for_execution(vae: Any, samples: torch.Tensor):
    model_management, comfy_na, comfy_kitchen = _require_comfy_runtime()
    patcher = getattr(vae, "patcher", None)
    if patcher is None:
        raise ValueError("Connected VAE does not expose its Comfy ModelPatcher.")
    # Use the exact VAE-side loading hook when available; current Comfy exposes
    # prepare_decode specifically for decode entry points that bypass VAE.decode().
    prepare_decode = getattr(vae, "prepare_decode", None)
    if callable(prepare_decode):
        prepare_decode(samples.shape)
    else:
        memory_used = int(vae.memory_used_decode(samples.shape, vae.vae_dtype))
        model_management.load_models_gpu(
            [patcher],
            memory_required=memory_used,
            force_full_load=bool(getattr(vae, "disable_offload", False)),
        )
    return model_management, comfy_na, comfy_kitchen


def execute_exact_deterministic_stages(
    vae: Any,
    final_video_latent: dict[str, Any],
    deterministic_context: DFRDeterministicDecoderKeyframeContext,
    exact_port: DFRExactDeterministicDualStreamPort,
    *,
    collect_diagnostics: bool = True,
) -> tuple[DFRDeterministicStageExecution, str]:
    if not isinstance(deterministic_context, DFRDeterministicDecoderKeyframeContext):
        raise ValueError(
            f"Expected DFRDeterministicDecoderKeyframeContext, got {type(deterministic_context).__name__}."
        )
    if not isinstance(exact_port, DFRExactDeterministicDualStreamPort):
        raise ValueError(
            f"Expected DFRExactDeterministicDualStreamPort, got {type(exact_port).__name__}."
        )
    if not isinstance(final_video_latent, dict) or not torch.is_tensor(final_video_latent.get("samples")):
        raise ValueError("final_video_latent must be a Comfy LATENT containing tensor 'samples'.")
    samples = final_video_latent["samples"]
    if samples.ndim != 5:
        raise ValueError(f"final_video_latent samples must be [B,C,T,H,W], got {tuple(samples.shape)}.")
    if int(samples.shape[1]) != int(deterministic_context.keyframe_channels):
        raise ValueError("Final video latent channel count disagrees with the validated decoder context.")
    if tuple(int(v) for v in samples.shape[-2:]) != tuple(
        int(v) for v in deterministic_context.typed_keyframe_latents.shape[-2:]
    ):
        raise ValueError("Final video/keyframe latent H/W must match before deterministic decode.")

    model_management, comfy_na, comfy_kitchen = _load_comfy_vae_for_execution(vae, samples)
    first_stage_model = vae.first_stage_model
    decoder = first_stage_model.decoder
    if bool(getattr(decoder, "deferred_stage4_upsample", False)):
        raise ValueError(
            "C24b targets current Comfy combined NADiffusionDecoder geometry; deferred stage-4 upsample is not active in the validated runtime."
        )

    device = vae.device
    dtype = vae.vae_dtype
    samples_dev = samples.to(device=device, dtype=dtype)
    typed_keyframes = deterministic_context.typed_keyframe_latents.to(device=device, dtype=dtype)
    pixel_indices = torch.tensor(
        deterministic_context.keyframe_pixel_frame_indices,
        dtype=torch.long,
        device="cpu",
    )
    valid = torch.tensor(
        deterministic_context.stage_descriptors[0].keyframe_valid_mask,
        dtype=torch.bool,
        device=device,
    )
    remaining = tuple(int(v) for v in deterministic_context.temporal_scale_schedule)

    # Source: trailing latent border workaround replicates the final VIDEO latent
    # frame through stages 1-4. Keyframe planes have no temporal extent and are not padded.
    ghost = int(getattr(decoder, "trailing_pad_latent_frames", 2))
    if ghost < 0:
        raise ValueError(f"Invalid decoder trailing_pad_latent_frames={ghost}.")
    if ghost:
        tail = samples_dev[:, :, -1:].expand(-1, -1, ghost, -1, -1)
        video_padded = torch.cat([samples_dev, tail], dim=2)
    else:
        video_padded = samples_dev

    if collect_diagnostics:
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = None
        if device.type == "cuda":
            cuda_rng_before = torch.cuda.get_rng_state(device).clone()
        params_before = _sample_parameter_checksum(decoder)

    stage_video_shapes: list[tuple[int, ...]] = []
    stage_keyframe_shapes: list[tuple[int, ...]] = []
    stage_times: list[tuple[float, ...]] = []
    stage_valid_counts: list[int] = []

    with model_management.cuda_device_context(device), torch.inference_mode():
        # Exact source ordering: video un-normalize -> shared conv_in.
        video_un = first_stage_model.per_channel_statistics.un_normalize(video_padded)
        video_x = decoder.conv_in(video_un.permute(0, 2, 3, 4, 1))

        # C22b context is exactly: keyframe un-normalize -> trained type_emb.
        keyframe_x = decoder.conv_in(typed_keyframes.permute(0, 2, 3, 4, 1))
        initial_times = keyframe_clip_times_official(
            pixel_indices,
            remaining[0],
            0,
        ).to(device=device)
        stream = _KeyframeStream(keyframe_x, initial_times, valid).masked()

        if collect_diagnostics:
            stage_video_shapes.append(tuple(int(v) for v in video_x.shape))
            stage_keyframe_shapes.append(tuple(int(v) for v in stream.x.shape))
            stage_times.append(tuple(float(v) for v in stream.times.detach().cpu().tolist()))
            stage_valid_counts.append(int(stream.valid.sum().item()))

        # Official forward_stages_1_to_3_with_keyframes: stages 0,1,2 full volume.
        for stage_i in range(3):
            video_x, stream = _run_det_stage_with_keyframes_exact(
                comfy_na,
                comfy_kitchen,
                decoder,
                video_x,
                stream,
                stage_i,
                pixel_indices,
                drop_leading_frame=True,
                next_time_origin=0.0,
                clip_start_frame=0,
                remaining_time_strides=remaining,
                num_slots=exact_port.num_slots,
            )
            if collect_diagnostics:
                stage_video_shapes.append(tuple(int(v) for v in video_x.shape))
                stage_keyframe_shapes.append(tuple(int(v) for v in stream.x.shape))
                stage_times.append(tuple(float(v) for v in stream.times.detach().cpu().tolist()))
                stage_valid_counts.append(int(stream.valid.sum().item()))

        if collect_diagnostics:
            # The legacy atomic executor proves the Stage-4 bridge with a real
            # origin probe plus a negative control.  Production immediately runs
            # the complete tiled Stage 4, so both extra forwards are redundant.
            stage4_kernel = tuple(int(v) for v in decoder.det_stages[3][0].attn.kernel_size)
            pt = min(int(video_x.shape[1]), max(1, stage4_kernel[0]))
            ph = min(int(video_x.shape[2]), max(1, stage4_kernel[1]))
            pw = min(int(video_x.shape[3]), max(1, stage4_kernel[2]))
            probe_video = video_x[:, :pt, :ph, :pw].clone()
            probe_stream = _KeyframeStream(
                x=stream.x[:, :, :ph, :pw].clone(),
                times=keyframe_clip_times_official(pixel_indices, remaining[3], 0).to(device=device),
                valid=stream.valid.clone(),
            )
            probe_video_out, probe_stream_out = _run_det_stage_with_keyframes_exact(
                comfy_na,
                comfy_kitchen,
                decoder,
                probe_video.clone(),
                probe_stream,
                3,
                pixel_indices,
                drop_leading_frame=True,
                next_time_origin=0.0,
                clip_start_frame=0,
                remaining_time_strides=remaining,
                num_slots=exact_port.num_slots,
            )

            invalid_stream = _KeyframeStream(
                x=torch.zeros_like(probe_stream.x),
                times=probe_stream.times,
                valid=torch.zeros_like(probe_stream.valid),
            )
            invalid_video_out, invalid_stream_out = _run_det_stage_with_keyframes_exact(
                comfy_na,
                comfy_kitchen,
                decoder,
                probe_video.clone(),
                invalid_stream,
                3,
                pixel_indices,
                drop_leading_frame=True,
                next_time_origin=0.0,
                clip_start_frame=0,
                remaining_time_strides=remaining,
                num_slots=exact_port.num_slots,
            )

    if collect_diagnostics:
        params_after = _sample_parameter_checksum(decoder)
        cpu_rng_after = torch.random.get_rng_state().clone()
        cuda_rng_after = None
        if device.type == "cuda":
            cuda_rng_after = torch.cuda.get_rng_state(device).clone()
        cross_stream = float(
            (probe_video_out.float() - invalid_video_out.float()).abs().max().item()
        )
        invalid_zero = float(invalid_stream_out.x.float().abs().max().item())
        probe_video_input_shape = tuple(int(v) for v in probe_video.shape)
        probe_keyframe_input_shape = tuple(int(v) for v in probe_stream.x.shape)
        probe_video_output_shape = tuple(int(v) for v in probe_video_out.shape)
        probe_keyframe_output_shape = tuple(int(v) for v in probe_stream_out.x.shape)
        probe_output_times = tuple(float(v) for v in probe_stream_out.times.detach().cpu().tolist())
        parameters_unchanged = params_before == params_after
        cpu_rng_unchanged = torch.equal(cpu_rng_before, cpu_rng_after)
        cuda_rng_unchanged = (
            True if device.type != "cuda" else torch.equal(cuda_rng_before, cuda_rng_after)
        )
    else:
        probe_video_input_shape = ()
        probe_keyframe_input_shape = ()
        probe_video_output_shape = ()
        probe_keyframe_output_shape = ()
        probe_output_times = ()
        cross_stream = 0.0
        invalid_zero = 0.0
        parameters_unchanged = True
        cpu_rng_unchanged = True
        cuda_rng_unchanged = True

    state = DFRDeterministicStageExecution(
        version=EXACT_DET_EXECUTION_VERSION,
        source_contract=exact_port.source_contract,
        execution_mode=(
            "official_full_stages_1_to_3_plus_real_stage4_origin_probe"
            if collect_diagnostics
            else "official_full_stages_1_to_3_production"
        ),
        device=str(device),
        dtype=str(dtype),
        input_video_shape=tuple(int(v) for v in samples.shape),
        ghost_pad_latent_frames=ghost,
        padded_video_shape=tuple(int(v) for v in video_padded.shape),
        stage_video_shapes=tuple(stage_video_shapes),
        stage_keyframe_shapes=tuple(stage_keyframe_shapes),
        stage_keyframe_times=tuple(stage_times),
        stage_keyframe_valid_counts=tuple(stage_valid_counts),
        stage4_input_video=video_x,
        stage4_input_keyframes=stream.x,
        stage4_input_times=stream.times,
        stage4_input_valid=stream.valid,
        stage4_probe_video_input_shape=probe_video_input_shape,
        stage4_probe_keyframe_input_shape=probe_keyframe_input_shape,
        stage4_probe_video_output_shape=probe_video_output_shape,
        stage4_probe_keyframe_output_shape=probe_keyframe_output_shape,
        stage4_probe_output_times=probe_output_times,
        stage4_probe_cross_stream_influence=cross_stream,
        stage4_probe_invalid_keyframe_zero_error=invalid_zero,
        decoder_parameters_unchanged=parameters_unchanged,
        cpu_rng_unchanged=cpu_rng_unchanged,
        cuda_rng_unchanged=cuda_rng_unchanged,
        stage5_called=False,
        decoder_mutated=False,
    )

    report = deterministic_stage_execution_report(state) if collect_diagnostics else ""
    return state, report


def _expected_full_stage_shapes(
    state: DFRDeterministicStageExecution,
    context: DFRDeterministicDecoderKeyframeContext,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    b, c, t, h, w = state.input_video_shape
    t = t + int(state.ghost_pad_latent_frames)
    channels = (2048, 1024, 512, 512)
    v_shapes: list[tuple[int, ...]] = [(b, t, h, w, channels[0])]
    k_shapes: list[tuple[int, ...]] = [
        (b, int(context.keyframe_count), h, w, channels[0])
    ]
    for i, (pt, ph, pw) in enumerate(context.upsample_strides[:3]):
        t = t * int(pt) - (1 if int(pt) == 2 else 0)
        h = h * int(ph)
        w = w * int(pw)
        v_shapes.append((b, t, h, w, channels[i + 1]))
        k_shapes.append((b, int(context.keyframe_count), h, w, channels[i + 1]))
    return tuple(v_shapes), tuple(k_shapes)


def validate_exact_deterministic_stages(
    execution: DFRDeterministicStageExecution,
    deterministic_context: DFRDeterministicDecoderKeyframeContext,
) -> dict[str, Any]:
    if not isinstance(execution, DFRDeterministicStageExecution):
        raise ValueError(
            f"Expected DFRDeterministicStageExecution, got {type(execution).__name__}."
        )
    if not isinstance(deterministic_context, DFRDeterministicDecoderKeyframeContext):
        raise ValueError(
            f"Expected DFRDeterministicDecoderKeyframeContext, got {type(deterministic_context).__name__}."
        )

    expected_v, expected_k = _expected_full_stage_shapes(execution, deterministic_context)
    geometry_error = 0.0 if execution.stage_video_shapes == expected_v and execution.stage_keyframe_shapes == expected_k else 1.0

    pixel_indices = torch.tensor(
        deterministic_context.keyframe_pixel_frame_indices,
        dtype=torch.long,
    )
    expected_times = tuple(
        tuple(
            float(v)
            for v in keyframe_clip_times_official(pixel_indices, stride, 0).tolist()
        )
        for stride in deterministic_context.temporal_scale_schedule[:4]
    )
    timing_error = 0.0 if execution.stage_keyframe_times == expected_times else 1.0
    plane_count_error = 0.0 if all(
        shape[1] == int(deterministic_context.keyframe_count)
        for shape in execution.stage_keyframe_shapes
    ) else 1.0
    valid_count_error = 0.0 if all(
        count == int(deterministic_context.keyframe_count)
        for count in execution.stage_keyframe_valid_counts
    ) else 1.0

    expected_stage4_times = tuple(
        float(v)
        for v in keyframe_clip_times_official(pixel_indices, deterministic_context.temporal_scale_schedule[4], 0).tolist()
    )
    stage4_time_error = 0.0 if execution.stage4_probe_output_times == expected_stage4_times else 1.0

    # The probe uses exactly stage4 kernel extents from the origin. Its upsample
    # shape is therefore source-defined by the validated fourth stride.
    pvin = execution.stage4_probe_video_input_shape
    pkin = execution.stage4_probe_keyframe_input_shape
    pt, ph, pw = deterministic_context.upsample_strides[3]
    expected_probe_v = (
        pvin[0],
        pvin[1] * pt - (1 if pt == 2 else 0),
        pvin[2] * ph,
        pvin[3] * pw,
        256,
    )
    expected_probe_k = (
        pkin[0],
        pkin[1],
        pkin[2] * ph,
        pkin[3] * pw,
        256,
    )
    stage4_geometry_error = 0.0 if (
        execution.stage4_probe_video_output_shape == expected_probe_v
        and execution.stage4_probe_keyframe_output_shape == expected_probe_k
    ) else 1.0

    passed = (
        geometry_error == 0.0
        and timing_error == 0.0
        and plane_count_error == 0.0
        and valid_count_error == 0.0
        and stage4_time_error == 0.0
        and stage4_geometry_error == 0.0
        and execution.stage4_probe_cross_stream_influence > 1e-7
        and execution.stage4_probe_invalid_keyframe_zero_error == 0.0
        and execution.decoder_parameters_unchanged
        and execution.cpu_rng_unchanged
        and execution.cuda_rng_unchanged
        and execution.stage5_called is False
        and execution.decoder_mutated is False
    )
    return {
        "passed": bool(passed),
        "geometry_error": geometry_error,
        "timing_error": timing_error,
        "plane_count_error": plane_count_error,
        "valid_count_error": valid_count_error,
        "stage4_time_error": stage4_time_error,
        "stage4_geometry_error": stage4_geometry_error,
        "stage4_probe_cross_stream_influence": float(execution.stage4_probe_cross_stream_influence),
        "stage4_probe_invalid_keyframe_zero_error": float(execution.stage4_probe_invalid_keyframe_zero_error),
        "decoder_parameters_unchanged": bool(execution.decoder_parameters_unchanged),
        "cpu_rng_unchanged": bool(execution.cpu_rng_unchanged),
        "cuda_rng_unchanged": bool(execution.cuda_rng_unchanged),
        "stage5_called": bool(execution.stage5_called),
        "stage_video_shapes": execution.stage_video_shapes,
        "stage_keyframe_shapes": execution.stage_keyframe_shapes,
        "stage_keyframe_times": execution.stage_keyframe_times,
        "stage4_probe_video_output_shape": execution.stage4_probe_video_output_shape,
        "stage4_probe_keyframe_output_shape": execution.stage4_probe_keyframe_output_shape,
        "stage4_probe_output_times": execution.stage4_probe_output_times,
    }


def deterministic_stage_execution_report(state: DFRDeterministicStageExecution) -> str:
    return (
        f"PASS={state.decoder_parameters_unchanged and state.cpu_rng_unchanged and state.cuda_rng_unchanged and state.stage4_probe_cross_stream_influence > 1e-7 and state.stage4_probe_invalid_keyframe_zero_error == 0.0}; "
        f"stage=U3.4b3_C24b_execute_exact_det; mode={state.execution_mode}; device={state.device}; dtype={state.dtype}; "
        f"input_video_shape={state.input_video_shape}; ghost_pad_latent_frames={state.ghost_pad_latent_frames}; "
        f"padded_video_shape={state.padded_video_shape}; stage_video_shapes={state.stage_video_shapes}; "
        f"stage_keyframe_shapes={state.stage_keyframe_shapes}; stage_keyframe_times={state.stage_keyframe_times}; "
        f"stage_keyframe_valid_counts={state.stage_keyframe_valid_counts}; "
        f"stage4_probe_video_input_shape={state.stage4_probe_video_input_shape}; "
        f"stage4_probe_keyframe_input_shape={state.stage4_probe_keyframe_input_shape}; "
        f"stage4_probe_video_output_shape={state.stage4_probe_video_output_shape}; "
        f"stage4_probe_keyframe_output_shape={state.stage4_probe_keyframe_output_shape}; "
        f"stage4_probe_output_times={state.stage4_probe_output_times}; "
        f"stage4_probe_cross_stream_influence={state.stage4_probe_cross_stream_influence:.9g}; "
        f"stage4_probe_invalid_keyframe_zero_error={state.stage4_probe_invalid_keyframe_zero_error:.9g}; "
        f"decoder_parameters_unchanged={state.decoder_parameters_unchanged}; cpu_rng_unchanged={state.cpu_rng_unchanged}; "
        f"cuda_rng_unchanged={state.cuda_rng_unchanged}; stage5_called=False; decoder_mutated=False."
    )
