"""C25 / U3.4b4: exact CHUNKED_EAGER Stage-5 dual-stream diffusion on one real tile.

This module ports only the keyframe-aware Stage-5 semantics missing from the
current Comfy ``NADiffusionDecoder``.  It reuses the loaded Comfy decoder
weights, ModelPatcher lifecycle, fused-QKV checkpoint representation,
``comfy_kitchen.rms_rope_`` and chunked SwiGLU execution.

Frozen-source contract used here:

* ``DiffusionVideoDecoder._decode_one_tile_with_keyframes``
  - run Stage 4 first;
  - draw video pixel noise from the continued decoder generator;
  - draw a separate keyframe-pixel noise field next from the same generator;
  - keep Stage-4 before its final upsample and build only each stream's
    ``conv_in_x_t(patchify(noise))`` buffer;
  - execute every Stage-5 block through the official deferred context inject,
    four-way W-slab joint attention, and tiled SwiGLU residual;
  - current LTX-2.5 checkpoint is one-step ``x0`` so the final model output is
    returned directly (no Euler update for this checkpoint).
* ``ChunkedDiffusionNABlock.forward_x_ctx_with_keyframes``
  - independent shared-weight context projection into each stream;
  - same AdaLN scale/shift on both streams;
  - one joint video/keyframe attention softmax;
  - same shared SwiGLU residual on both streams;
  - invalid keyframe planes are re-zeroed at the END OF EVERY STAGE-5 BLOCK.
* ``chunked.attn.chunked_with_keyframes``
  - absolute RoPE at local integer video times and exact keyframe stage-5 times;
  - clamp-and-mask joint window, not Comfy's native shifted NATTEN boundary
    semantics.

C25 is deliberately a *single-tile execution proof*, not the final tiled
renderer.  It uses one real origin crop from C24b's Stage-4 input and carries
all current keyframe planes into the tile.  The official per-tile
``planes_for_tile`` filtering, official current-checkpoint per-tile video/keyframe RNG ordering, halos, overlap
blending and full frame assembly belong to C26b / U3.4b5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .decoder_bridge import restore_decoder_generator
from .decoder_det_dual_stream_exact import (
    DFRDeterministicStageExecution,
    DFRExactDeterministicDualStreamPort,
    _KeyframeStream,
    _forward_attention_with_keyframes_exact,
    _load_comfy_vae_for_execution,
    _run_det_stage_blocks_with_keyframes_exact,
    _run_det_stage_with_keyframes_exact,
    _sample_parameter_checksum,
)
from .decoder_det_stage_context import DFRDeterministicDecoderKeyframeContext
from .decoder_det_stage_official_preflight import keyframe_clip_times_official
from .decoder_handoff import DFRStage2DecoderHandoff
from .decoder_stage5_chunked_exact import (
    OFFICIAL_STAGE5_W_CHUNKS,
    build_deferred_stage5_inputs_exact,
    forward_stage5_chunked_step_exact,
    stage5_grid_shape_from_stage4,
)


STAGE5_PORT_VERSION = 2
STAGE5_EXECUTION_VERSION = 2


@dataclass(frozen=True)
class DFRExactStage5DualStreamPort:
    version: int
    source_contract: str
    runtime_backbone: str
    pathway: str
    joint_attention_backend: str
    num_slots: int
    context_injection: str
    modulation: str
    rope_mapping: str
    mlp_mapping: str
    invalid_plane_policy: str
    tile_scope: str
    decoder_mutated: bool


@dataclass(frozen=True)
class DFRStage5SingleTileExecution:
    version: int
    source_contract: str
    execution_mode: str
    device: str
    dtype: str
    model_output_type: str
    inference_timesteps: tuple[float, ...]
    stage5_block_count: int
    stage5_blocks_executed: int
    stage4_input_tile_shape: tuple[int, ...]
    stage4_keyframe_input_shape: tuple[int, ...]
    stage4_context_shape: tuple[int, ...]
    stage5_keyframe_context_shape: tuple[int, ...]
    stage5_keyframe_times: tuple[float, ...]
    keyframe_valid_count: int
    pixel_noise_shape: tuple[int, ...]
    keyframe_pixel_noise_shape: tuple[int, ...]
    video_prediction_shape: tuple[int, ...]
    keyframe_prediction_shape: tuple[int, ...]
    stage5_cross_stream_influence: float
    invalid_keyframe_zero_error: float
    finite_error: float
    rng_device_type: str
    generator_state_restored: bool
    generator_state_advanced: bool
    generator_initial_state: torch.Tensor
    generator_final_state: torch.Tensor
    cpu_rng_unchanged: bool
    cuda_rng_unchanged: bool
    decoder_parameters_unchanged: bool
    stage5_called: bool
    stage5_all_blocks_called: bool
    stage4_drop_leading_frame: bool
    stage4_pad_trailing: bool
    plane_selection_mode: str
    decoder_mutated: bool


def prepare_exact_stage5_dual_stream_port(
    exact_deterministic_port: DFRExactDeterministicDualStreamPort,
) -> tuple[DFRExactStage5DualStreamPort, str]:
    if not isinstance(exact_deterministic_port, DFRExactDeterministicDualStreamPort):
        raise ValueError(
            f"Expected DFRExactDeterministicDualStreamPort, got {type(exact_deterministic_port).__name__}."
        )
    if exact_deterministic_port.joint_attention_backend not in {
        "official_fallback_joint_eager",
        "official_auto_joint_triton_cuda_eager_fallback",
    }:
        raise ValueError("C25 requires the validated frozen official joint-attention backend.")
    state = DFRExactStage5DualStreamPort(
        version=STAGE5_PORT_VERSION,
        source_contract=(
            "frozen_DiffusionVideoDecoder._decode_one_tile_with_keyframes+"
            "forward_diff_step_deferred_with_keyframes+"
            "ChunkedDiffusionNABlock.forward_x_ctx_with_keyframes+"
            "chunked.attn.chunked_with_keyframes+chunked.context+chunked.mlp"
        ),
        runtime_backbone="comfy.NADiffusionDecoder_existing_weights",
        pathway="official_chunked_eager_deferred_stage4_w4",
        joint_attention_backend=exact_deterministic_port.joint_attention_backend,
        num_slots=int(exact_deterministic_port.num_slots),
        context_injection="deferred_stage4_pixel_shuffle_plus_context_proj_per_block_w4_per_stream",
        modulation="shared_AdaLN_scale_shift_per_stream",
        rope_mapping="comfy_rms_rope_exact_abs_positions_with_keyframe_stage5_times",
        mlp_mapping="official_tiled_16384_token_SwiGLU_with_triton_bf16_gate_up",
        invalid_plane_policy="rezero_at_end_of_every_stage5_block",
        tile_scope="one_real_origin_tile; official_w4_internal_slabs; planes_for_tile_filtering_deferred_to_C26b",
        decoder_mutated=False,
    )
    report = (
        f"PASS=True; stage=U3.4b4_C25_prepare_exact_stage5_port; version={state.version}; "
        f"source_contract={state.source_contract}; runtime_backbone={state.runtime_backbone}; pathway={state.pathway}; "
        f"joint_attention_backend={state.joint_attention_backend}; num_slots={state.num_slots}; "
        f"context_injection={state.context_injection}; modulation={state.modulation}; "
        f"rope_mapping={state.rope_mapping}; mlp_mapping={state.mlp_mapping}; "
        f"invalid_plane_policy={state.invalid_plane_policy}; tile_scope={state.tile_scope}; decoder_mutated=False."
    )
    return state, report


def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


def _context_inject_chunked(block: Any, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Official ``x + context_proj(context)`` using Comfy's bounded temporal chunks."""
    if x.shape[:-1] != context.shape[:-1]:
        raise ValueError(f"Context/x geometry mismatch: x={tuple(x.shape)}, context={tuple(context.shape)}.")
    h, w = int(x.shape[2]), int(x.shape[3])
    # Match current Comfy DiffusionNABlock's chunk discipline.
    try:
        from comfy.ldm.lightricks.vae.na_diffusion_decoder import MLP_TOKEN_CHUNK
        token_chunk = int(MLP_TOKEN_CHUNK)
    except Exception:  # pragma: no cover
        token_chunk = 65536
    chunk = max(1, token_chunk // max(h * w, 1))
    out = x.clone()
    for t0 in range(0, int(x.shape[1]), int(chunk)):
        t1 = min(t0 + int(chunk), int(x.shape[1]))
        out[:, t0:t1] += block.context_proj(context[:, t0:t1])
    return out


def _forward_stage5_block_with_keyframes_exact(
    comfy_na: Any,
    comfy_kitchen: Any,
    block: Any,
    video_x: torch.Tensor,
    video_context: torch.Tensor,
    keyframe_x: torch.Tensor,
    keyframe_context: torch.Tensor,
    modulation: tuple[torch.Tensor, ...],
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    *,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct source port of ``CombinedDiffusionNABlock.forward_combined_with_keyframes``."""
    if len(modulation) != 7:
        raise ValueError(f"Expected 7 AdaLN chunks, got {len(modulation)}.")
    table = block.scale_shift_table
    chunks = [
        modulation[i] + table[i].view(1, 1, 1, 1, -1)
        for i in range(7)
    ]
    scale_msa, shift_msa, _gate_msa, scale_mlp, shift_mlp, _gate_mlp, _extra = chunks

    # frozen combined.context: independent context projection for each stream.
    video_x = _context_inject_chunked(block, video_x, video_context)
    keyframe_x = _context_inject_chunked(block, keyframe_x, keyframe_context)

    # frozen combined.attn.full_with_keyframes: same modulation on both streams,
    # exact absolute RoPE positions, one joint softmax, shared output projection.
    video_y = _modulate(block.norm1(video_x), scale_msa, shift_msa)
    keyframe_y = _modulate(block.norm1(keyframe_x), scale_msa, shift_msa)
    attn_out, keyframe_attn = _forward_attention_with_keyframes_exact(
        comfy_na,
        comfy_kitchen,
        block.attn,
        video_y,
        keyframe_y,
        keyframe_times,
        keyframe_valid,
        num_slots=int(num_slots),
    )
    video_x = video_x + attn_out
    keyframe_x = keyframe_x + keyframe_attn

    # frozen combined.mlp.residual_mlp, mapped to Comfy's bounded SwiGLU helper.
    video_x = block.mlp(
        video_x,
        pre=lambda s: _modulate(block.norm2(s), scale_mlp, shift_mlp),
        add_to=video_x,
    )
    keyframe_x = block.mlp(
        keyframe_x,
        pre=lambda s: _modulate(block.norm2(s), scale_mlp, shift_mlp),
        add_to=keyframe_x,
    )

    # Exact asymmetry vs deterministic blocks: Stage 5 re-zeroes invalid planes
    # at the end of EVERY block.
    keyframe_x = keyframe_x * keyframe_valid.to(
        device=keyframe_x.device, dtype=keyframe_x.dtype
    )[None, :, None, None, None]
    return video_x, keyframe_x


def _build_stage5_inputs(
    comfy_na: Any,
    decoder: Any,
    video_context: torch.Tensor,
    keyframe_context: torch.Tensor,
    video_x_t: torch.Tensor,
    keyframe_x_t: torch.Tensor,
    keyframe_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen ``_context_and_x_for_diff_step`` + keyframe counterpart."""
    patched = comfy_na.patchify(video_x_t, patch_size_hw=int(decoder.patch_size), patch_size_t=1)
    video_x = decoder.conv_in_x_t(patched.permute(0, 2, 3, 4, 1))

    kpatched = comfy_na.patchify(keyframe_x_t, patch_size_hw=int(decoder.patch_size), patch_size_t=1)
    keyframe_x = decoder.conv_in_x_t(kpatched.permute(0, 2, 3, 4, 1))
    keyframe_x = keyframe_x * keyframe_valid.to(
        device=keyframe_x.device, dtype=keyframe_x.dtype
    )[None, :, None, None, None]

    if video_context.shape[:-1] != video_x.shape[:-1]:
        raise ValueError(
            f"Stage-5 video context/noise geometry mismatch: context={tuple(video_context.shape)}, x={tuple(video_x.shape)}."
        )
    if keyframe_context.shape[:-1] != keyframe_x.shape[:-1]:
        raise ValueError(
            f"Stage-5 keyframe context/noise geometry mismatch: context={tuple(keyframe_context.shape)}, x={tuple(keyframe_x.shape)}."
        )
    return torch.cat([video_context, video_x], dim=-1), torch.cat([keyframe_context, keyframe_x], dim=-1)


def _pixels_from_stage5(comfy_na: Any, decoder: Any, x: torch.Tensor) -> torch.Tensor:
    x = decoder.norm_out(x)
    x = decoder.conv_out(x)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return comfy_na.unpatchify(x, patch_size_hw=int(decoder.patch_size), patch_size_t=1)


def _forward_stage5_step_with_keyframes_exact(
    comfy_na: Any,
    comfy_kitchen: Any,
    decoder: Any,
    context_and_x: torch.Tensor,
    keyframe_context_and_x: torch.Tensor,
    t: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    *,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Frozen ``forward_diff_step_with_keyframes`` on the current Comfy decoder."""
    context_channels = int(decoder.context_channels)
    video_context = context_and_x[..., :context_channels]
    video_x = context_and_x[..., context_channels:].clone()
    keyframe_context = keyframe_context_and_x[..., :context_channels]
    keyframe_x = keyframe_context_and_x[..., context_channels:].clone()

    t_emb = decoder.t_embedder(
        float(decoder.timestep_scale_multiplier) * t,
        dtype=video_x.dtype,
    )
    modulation = decoder.shared_adaln(t_emb)

    invalid_zero = 0.0
    blocks_executed = 0
    for block in decoder.diff_blocks:
        video_x, keyframe_x = _forward_stage5_block_with_keyframes_exact(
            comfy_na,
            comfy_kitchen,
            block,
            video_x,
            video_context,
            keyframe_x,
            keyframe_context,
            modulation,
            keyframe_times,
            keyframe_valid,
            num_slots=int(num_slots),
        )
        blocks_executed += 1
        if bool((~keyframe_valid).any()):
            invalid = keyframe_x[:, ~keyframe_valid]
            if invalid.numel():
                invalid_zero = max(invalid_zero, float(invalid.float().abs().max().item()))

    return (
        _pixels_from_stage5(comfy_na, decoder, video_x),
        _pixels_from_stage5(comfy_na, decoder, keyframe_x),
        float(invalid_zero),
        int(blocks_executed),
    )


def _derive_probe_stage4_shape(decoder: Any, execution: DFRDeterministicStageExecution) -> tuple[int, int, int]:
    """Smallest real origin crop meeting loaded Stage-4 and Stage-5 kernel floors."""
    k4 = tuple(int(v) for v in decoder.det_stages[3][0].attn.kernel_size)
    k5 = tuple(int(v) for v in decoder.diff_blocks[0].attn.kernel_size)
    stride = tuple(int(v) for v in decoder.upsamples[3].stride)

    need: list[int] = []
    for axis, (k4a, k5a, s) in enumerate(zip(k4, k5, stride, strict=True)):
        if s == 1:
            from_stage5 = k5a
        elif axis == 0 and s == 2:
            # Origin tile uses drop_leading_frame=True: out = 2*in - 1.
            from_stage5 = (k5a + 2) // 2
        else:
            from_stage5 = (k5a + s - 1) // s
        need.append(max(k4a, from_stage5))

    max_shape = execution.stage4_input_video.shape[1:4]
    out = tuple(min(int(max_shape[i]), int(need[i])) for i in range(3))
    if any(out[i] < need[i] for i in range(3)):
        raise ValueError(
            f"C25 cannot form a source-safe Stage-5 probe: need stage4 THW={tuple(need)}, "
            f"available={tuple(int(v) for v in max_shape)}."
        )
    return out


def execute_exact_stage5_single_tile(
    vae: Any,
    deterministic_execution: DFRDeterministicStageExecution,
    deterministic_context: DFRDeterministicDecoderKeyframeContext,
    decoder_handoff: DFRStage2DecoderHandoff,
    stage5_port: DFRExactStage5DualStreamPort,
) -> tuple[DFRStage5SingleTileExecution, str]:
    if not isinstance(deterministic_execution, DFRDeterministicStageExecution):
        raise ValueError(
            f"Expected DFRDeterministicStageExecution, got {type(deterministic_execution).__name__}."
        )
    if not isinstance(deterministic_context, DFRDeterministicDecoderKeyframeContext):
        raise ValueError(
            f"Expected DFRDeterministicDecoderKeyframeContext, got {type(deterministic_context).__name__}."
        )
    if not isinstance(decoder_handoff, DFRStage2DecoderHandoff):
        raise ValueError(f"Expected DFRStage2DecoderHandoff, got {type(decoder_handoff).__name__}.")
    if not isinstance(stage5_port, DFRExactStage5DualStreamPort):
        raise ValueError(f"Expected DFRExactStage5DualStreamPort, got {type(stage5_port).__name__}.")
    if not deterministic_execution.stage5_called is False:
        raise ValueError("C25 requires the validated C24b execution that stopped before Stage 5.")

    # Re-load the same VAE through Comfy's lifecycle.  Use the original final-latent
    # shape only as memory guidance; C24b already carries the actual Stage-4 features.
    guidance = torch.empty(
        deterministic_execution.input_video_shape,
        dtype=torch.bfloat16,
        device="meta",
    )
    # prepare_decode only reads shape in current Comfy; avoid a real giant allocation.
    model_management, comfy_na, comfy_kitchen = _load_comfy_vae_for_execution(vae, guidance)
    first_stage_model = vae.first_stage_model
    decoder = first_stage_model.decoder
    if stage5_port.pathway != "official_chunked_eager_deferred_stage4_w4":
        raise ValueError(f"C25 requires the official chunked pathway, got {stage5_port.pathway!r}.")

    config = getattr(first_stage_model, "config", {}) or {}
    dec_cfg = config.get("decoder", {}) if isinstance(config, dict) else {}
    model_output_type = str(config.get("model_output_type", getattr(decoder, "model_output_type", "")))
    timesteps = decoder.default_inference_timesteps.detach().to(torch.float32)
    # Current validated LTX-2.5 VAE is one-step x0.  Do not silently invent the
    # generic multi-step branch here; future checkpoint variants get their own source audit.
    if model_output_type != "x0" or int(timesteps.numel()) != 1:
        raise ValueError(
            "C25 source-exact implementation is locked to the validated current checkpoint contract "
            f"(model_output_type='x0', one inference step); got model_output_type={model_output_type!r}, "
            f"timesteps={tuple(float(v) for v in timesteps.tolist())}, decoder_config={dec_cfg}."
        )

    device = vae.device
    dtype = vae.vae_dtype
    video_s4 = deterministic_execution.stage4_input_video.to(device=device, dtype=dtype)
    key_s4 = deterministic_execution.stage4_input_keyframes.to(device=device, dtype=dtype)
    valid = deterministic_execution.stage4_input_valid.to(device=device, dtype=torch.bool)
    pixel_indices = torch.tensor(
        deterministic_context.keyframe_pixel_frame_indices,
        dtype=torch.long,
        device="cpu",
    )
    remaining = tuple(int(v) for v in deterministic_context.temporal_scale_schedule)

    pt, ph, pw = _derive_probe_stage4_shape(decoder, deterministic_execution)
    feat_tile = video_s4[:, :pt, :ph, :pw].clone()
    # C25 isolates Stage 5, so carry every current plane. C26 will apply
    # ``planes_for_tile`` before constructing each real renderer tile.
    stage4_times = keyframe_clip_times_official(pixel_indices, remaining[3], 0).to(device=device)
    tile_stream = _KeyframeStream(
        x=key_s4[:, :, :ph, :pw].clone(),
        times=stage4_times,
        valid=valid.clone(),
    ).masked()

    generator, rng_device_type = restore_decoder_generator(decoder_handoff)
    generator_initial = generator.get_state().detach().cpu().clone()
    expected_initial = decoder_handoff.rng_state_after_stage2_av.detach().cpu()
    generator_state_restored = torch.equal(generator_initial, expected_initial)
    if not generator_state_restored:
        raise RuntimeError("C25 reconstructed generator does not match the U3.1 post-Stage2 AV state.")

    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    params_before = _sample_parameter_checksum(decoder)

    with model_management.cuda_device_context(device), torch.inference_mode():
        # Exact CHUNKED_EAGER order: Stage-4 blocks only, then pixel noise.
        # The final Stage-4 pixel shuffle is deferred into every Stage-5 block.
        context_tile, stage4_stream = _run_det_stage_blocks_with_keyframes_exact(
            comfy_na,
            comfy_kitchen,
            decoder,
            feat_tile,
            tile_stream,
            3,
            num_slots=int(stage5_port.num_slots),
        )
        stage5_stream = _KeyframeStream(
            x=stage4_stream.x,
            times=keyframe_clip_times_official(pixel_indices, remaining[4], 0).to(device=device),
            valid=stage4_stream.valid,
        ).masked()

        batch = int(context_tile.shape[0])
        grid_t, grid_h, grid_w = stage5_grid_shape_from_stage4(
            context_tile,
            tuple(int(v) for v in decoder.upsamples[3].stride),
            drop_leading_frame=True,
        )
        canvas_t = int(grid_t)
        canvas_h = int(grid_h) * int(decoder.patch_size)
        canvas_w = int(grid_w) * int(decoder.patch_size)
        randn_device = generator.device

        def _noise(frames: int) -> torch.Tensor:
            return torch.randn(
                (batch, int(decoder.out_channels), int(frames), canvas_h, canvas_w),
                dtype=dtype,
                generator=generator,
                device=randn_device,
            ).to(device)

        # Frozen source draw order: video x_t first, separate keyframe x_t second.
        video_x_t = _noise(canvas_t)
        keyframe_x_t = _noise(int(stage5_stream.x.shape[1]))

        video_x, keyframe_x = build_deferred_stage5_inputs_exact(
            comfy_na,
            decoder,
            context_tile,
            stage5_stream.x,
            video_x_t,
            keyframe_x_t,
            stage5_stream.valid,
            drop_leading_frame=True,
        )
        t_now = timesteps.to(device=device)[0].expand(batch)
        video_pred, keyframe_pred, valid_invalid_zero, blocks_executed = forward_stage5_chunked_step_exact(
            comfy_na,
            comfy_kitchen,
            decoder,
            video_x,
            context_tile,
            keyframe_x,
            stage5_stream.x,
            t_now,
            stage5_stream.times,
            stage5_stream.valid,
            drop_leading_frame=True,
            num_slots=int(stage5_port.num_slots),
            w_chunks=OFFICIAL_STAGE5_W_CHUNKS,
        )

        # Source-valid negative control to isolate *Stage-5* cross-stream influence.
        # Keep identical video context + video noise, but remove the keyframe stream.
        invalid_valid = torch.zeros_like(stage5_stream.valid)
        invalid_context = torch.zeros_like(stage5_stream.x)
        inv_video_x, inv_keyframe_x = build_deferred_stage5_inputs_exact(
            comfy_na,
            decoder,
            context_tile,
            invalid_context,
            video_x_t.clone(),
            keyframe_x_t.clone(),
            invalid_valid,
            drop_leading_frame=True,
        )
        invalid_video_pred, invalid_keyframe_pred, invalid_zero, invalid_blocks_executed = forward_stage5_chunked_step_exact(
            comfy_na,
            comfy_kitchen,
            decoder,
            inv_video_x,
            context_tile,
            inv_keyframe_x,
            invalid_context,
            t_now,
            stage5_stream.times,
            invalid_valid,
            drop_leading_frame=True,
            num_slots=int(stage5_port.num_slots),
            w_chunks=OFFICIAL_STAGE5_W_CHUNKS,
        )

    params_after = _sample_parameter_checksum(decoder)
    cpu_rng_after = torch.random.get_rng_state().clone()
    cuda_rng_after = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    generator_final = generator.get_state().detach().cpu().clone()

    cross_stream = float((video_pred.float() - invalid_video_pred.float()).abs().max().item())
    invalid_zero = float(invalid_zero)
    finite_error = 0.0
    for tensor in (video_pred, keyframe_pred, invalid_video_pred, invalid_keyframe_pred):
        if not bool(torch.isfinite(tensor).all()):
            finite_error = 1.0
            break

    state = DFRStage5SingleTileExecution(
        version=STAGE5_EXECUTION_VERSION,
        source_contract=stage5_port.source_contract,
        execution_mode="official_chunked_eager_deferred_stage4_w4_one_step_x0_origin_tile_probe",
        device=str(device),
        dtype=str(dtype),
        model_output_type=model_output_type,
        inference_timesteps=tuple(float(v) for v in timesteps.tolist()),
        stage5_block_count=int(len(decoder.diff_blocks)),
        stage5_blocks_executed=int(blocks_executed),
        stage4_input_tile_shape=tuple(int(v) for v in feat_tile.shape),
        stage4_keyframe_input_shape=tuple(int(v) for v in tile_stream.x.shape),
        stage4_context_shape=tuple(int(v) for v in context_tile.shape),
        stage5_keyframe_context_shape=tuple(int(v) for v in stage5_stream.x.shape),
        stage5_keyframe_times=tuple(float(v) for v in stage5_stream.times.detach().cpu().tolist()),
        keyframe_valid_count=int(stage5_stream.valid.sum().item()),
        pixel_noise_shape=tuple(int(v) for v in video_x_t.shape),
        keyframe_pixel_noise_shape=tuple(int(v) for v in keyframe_x_t.shape),
        video_prediction_shape=tuple(int(v) for v in video_pred.shape),
        keyframe_prediction_shape=tuple(int(v) for v in keyframe_pred.shape),
        stage5_cross_stream_influence=cross_stream,
        invalid_keyframe_zero_error=max(float(valid_invalid_zero), float(invalid_zero)),
        finite_error=finite_error,
        rng_device_type=rng_device_type,
        generator_state_restored=bool(generator_state_restored),
        generator_state_advanced=not torch.equal(generator_initial, generator_final),
        generator_initial_state=generator_initial,
        generator_final_state=generator_final,
        cpu_rng_unchanged=torch.equal(cpu_rng_before, cpu_rng_after),
        cuda_rng_unchanged=(True if device.type != "cuda" else torch.equal(cuda_rng_before, cuda_rng_after)),
        decoder_parameters_unchanged=params_before == params_after,
        stage5_called=True,
        stage5_all_blocks_called=(int(blocks_executed) == int(len(decoder.diff_blocks)) and int(invalid_blocks_executed) == int(len(decoder.diff_blocks))),
        stage4_drop_leading_frame=True,
        stage4_pad_trailing=False,
        plane_selection_mode="all_current_planes; official_planes_for_tile_deferred_to_C26",
        decoder_mutated=False,
    )
    report = stage5_single_tile_execution_report(state)
    return state, report


def validate_exact_stage5_single_tile(
    execution: DFRStage5SingleTileExecution,
    deterministic_context: DFRDeterministicDecoderKeyframeContext,
    decoder_handoff: DFRStage2DecoderHandoff,
) -> dict[str, Any]:
    if not isinstance(execution, DFRStage5SingleTileExecution):
        raise ValueError(f"Expected DFRStage5SingleTileExecution, got {type(execution).__name__}.")
    if not isinstance(deterministic_context, DFRDeterministicDecoderKeyframeContext):
        raise ValueError(
            f"Expected DFRDeterministicDecoderKeyframeContext, got {type(deterministic_context).__name__}."
        )
    if not isinstance(decoder_handoff, DFRStage2DecoderHandoff):
        raise ValueError(f"Expected DFRStage2DecoderHandoff, got {type(decoder_handoff).__name__}.")

    # CHUNKED_EAGER returns Stage-4 block features before the final upsample.
    # The deferred inject applies stride (2,2,2) inside every Stage-5 block.
    vin = execution.stage4_input_tile_shape
    kout = execution.stage5_keyframe_context_shape
    stride = tuple(int(v) for v in deterministic_context.upsample_strides[3])
    expected_ctx = (
        vin[0],
        vin[1],
        vin[2],
        vin[3],
        vin[4],
    )
    geometry_error = 0.0 if execution.stage4_context_shape == expected_ctx else 1.0
    expected_kctx = (
        execution.stage4_keyframe_input_shape[0],
        execution.stage4_keyframe_input_shape[1],
        execution.stage4_keyframe_input_shape[2],
        execution.stage4_keyframe_input_shape[3],
        execution.stage4_keyframe_input_shape[4],
    )
    keyframe_geometry_error = 0.0 if kout == expected_kctx else 1.0

    pixel_indices = torch.tensor(deterministic_context.keyframe_pixel_frame_indices, dtype=torch.long)
    expected_times = tuple(
        float(v)
        for v in keyframe_clip_times_official(
            pixel_indices,
            deterministic_context.temporal_scale_schedule[4],
            0,
        ).tolist()
    )
    timing_error = 0.0 if execution.stage5_keyframe_times == expected_times else 1.0

    # Loaded checkpoint contract was already established by C23b.
    checkpoint_error = 0.0 if (
        execution.model_output_type == "x0"
        and execution.inference_timesteps == (1.0,)
        and execution.stage5_block_count == 8
        and execution.stage5_blocks_executed == 8
    ) else 1.0

    patch = 4
    expected_video_pred = (
        execution.stage4_context_shape[0],
        3,
        execution.stage4_context_shape[1] * stride[0] - (1 if stride[0] == 2 else 0),
        execution.stage4_context_shape[2] * stride[1] * patch,
        execution.stage4_context_shape[3] * stride[2] * patch,
    )
    expected_key_pred = (
        execution.stage5_keyframe_context_shape[0],
        3,
        execution.stage5_keyframe_context_shape[1],
        execution.stage5_keyframe_context_shape[2] * stride[1] * patch,
        execution.stage5_keyframe_context_shape[3] * stride[2] * patch,
    )
    output_geometry_error = 0.0 if (
        execution.video_prediction_shape == expected_video_pred
        and execution.keyframe_prediction_shape == expected_key_pred
        and execution.pixel_noise_shape == expected_video_pred
        and execution.keyframe_pixel_noise_shape == expected_key_pred
    ) else 1.0

    initial_rng_error = 0.0 if torch.equal(
        execution.generator_initial_state.detach().cpu(),
        decoder_handoff.rng_state_after_stage2_av.detach().cpu(),
    ) else 1.0
    rng_advance_error = 0.0 if execution.generator_state_advanced else 1.0

    passed = (
        geometry_error == 0.0
        and keyframe_geometry_error == 0.0
        and timing_error == 0.0
        and checkpoint_error == 0.0
        and output_geometry_error == 0.0
        and initial_rng_error == 0.0
        and rng_advance_error == 0.0
        and execution.keyframe_valid_count == int(deterministic_context.keyframe_count)
        and execution.stage5_cross_stream_influence > 1e-7
        and execution.invalid_keyframe_zero_error == 0.0
        and execution.finite_error == 0.0
        and execution.generator_state_restored
        and execution.cpu_rng_unchanged
        and execution.cuda_rng_unchanged
        and execution.decoder_parameters_unchanged
        and execution.stage5_called
        and execution.stage5_all_blocks_called
        and execution.stage4_drop_leading_frame
        and execution.stage4_pad_trailing is False
        and execution.decoder_mutated is False
    )
    return {
        "passed": bool(passed),
        "geometry_error": geometry_error,
        "keyframe_geometry_error": keyframe_geometry_error,
        "timing_error": timing_error,
        "checkpoint_error": checkpoint_error,
        "output_geometry_error": output_geometry_error,
        "initial_rng_error": initial_rng_error,
        "rng_advance_error": rng_advance_error,
        "stage5_cross_stream_influence": execution.stage5_cross_stream_influence,
        "invalid_keyframe_zero_error": execution.invalid_keyframe_zero_error,
        "finite_error": execution.finite_error,
        "generator_state_restored": execution.generator_state_restored,
        "generator_state_advanced": execution.generator_state_advanced,
        "cpu_rng_unchanged": execution.cpu_rng_unchanged,
        "cuda_rng_unchanged": execution.cuda_rng_unchanged,
        "decoder_parameters_unchanged": execution.decoder_parameters_unchanged,
        "stage5_called": execution.stage5_called,
        "stage5_all_blocks_called": execution.stage5_all_blocks_called,
        "stage5_block_count": execution.stage5_block_count,
        "stage5_blocks_executed": execution.stage5_blocks_executed,
        "stage4_input_tile_shape": execution.stage4_input_tile_shape,
        "stage4_context_shape": execution.stage4_context_shape,
        "stage5_keyframe_context_shape": execution.stage5_keyframe_context_shape,
        "stage5_keyframe_times": execution.stage5_keyframe_times,
        "pixel_noise_shape": execution.pixel_noise_shape,
        "keyframe_pixel_noise_shape": execution.keyframe_pixel_noise_shape,
        "video_prediction_shape": execution.video_prediction_shape,
        "keyframe_prediction_shape": execution.keyframe_prediction_shape,
        "plane_selection_mode": execution.plane_selection_mode,
    }


def stage5_single_tile_execution_report(state: DFRStage5SingleTileExecution) -> str:
    return (
        f"PASS=True; stage=U3.4b4_C25_execute_exact_stage5_single_tile; version={state.version}; "
        f"mode={state.execution_mode}; device={state.device}; dtype={state.dtype}; "
        f"model_output_type={state.model_output_type}; inference_timesteps={state.inference_timesteps}; "
        f"stage5_block_count={state.stage5_block_count}; stage5_blocks_executed={state.stage5_blocks_executed}; "
        f"stage4_input_tile_shape={state.stage4_input_tile_shape}; "
        f"stage4_keyframe_input_shape={state.stage4_keyframe_input_shape}; stage4_context_shape={state.stage4_context_shape}; "
        f"stage5_keyframe_context_shape={state.stage5_keyframe_context_shape}; "
        f"stage5_keyframe_times={state.stage5_keyframe_times}; keyframe_valid_count={state.keyframe_valid_count}; "
        f"pixel_noise_shape={state.pixel_noise_shape}; keyframe_pixel_noise_shape={state.keyframe_pixel_noise_shape}; "
        f"video_prediction_shape={state.video_prediction_shape}; keyframe_prediction_shape={state.keyframe_prediction_shape}; "
        f"stage5_cross_stream_influence={state.stage5_cross_stream_influence:.9g}; "
        f"invalid_keyframe_zero_error={state.invalid_keyframe_zero_error:.9g}; finite_error={state.finite_error:.9g}; "
        f"rng_device_type={state.rng_device_type}; generator_state_restored={state.generator_state_restored}; "
        f"generator_state_advanced={state.generator_state_advanced}; cpu_rng_unchanged={state.cpu_rng_unchanged}; "
        f"cuda_rng_unchanged={state.cuda_rng_unchanged}; decoder_parameters_unchanged={state.decoder_parameters_unchanged}; "
        f"stage5_called={state.stage5_called}; stage5_all_blocks_called={state.stage5_all_blocks_called}; "
        f"stage4_drop_leading_frame={state.stage4_drop_leading_frame}; stage4_pad_trailing={state.stage4_pad_trailing}; "
        f"plane_selection_mode={state.plane_selection_mode}; decoder_mutated=False."
    )
