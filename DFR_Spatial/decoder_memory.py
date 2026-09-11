"""Release sampling model residency before allocating DiffVAE features."""
import logging

import torch


def prepare_decoder_memory(vae):
    patcher = getattr(vae, "patcher", None)
    device = getattr(vae, "device", getattr(patcher, "load_device", None))
    if device is None or torch.device(device).type != "cuda":
        return
    import comfy.model_management as management

    device = torch.device(device)
    # free_memory expects LoadedModel entries, not ModelPatcher objects.
    # Preserve the VAE and any already-loaded wrappers sharing its base model.
    first_stage = getattr(vae, "first_stage_model", None)
    keep = [loaded for loaded in management.current_loaded_models
            if loaded.model is patcher or (
                first_stage is not None and getattr(loaded.model, "model", None) is first_stage)]
    before = management.get_free_memory(device)
    # Request full eviction of other managed models on this device. They remain
    # valid in the graph and Comfy can reload them for later sampling nodes.
    unloaded = management.free_memory(float("inf"), device, keep_loaded=keep, for_dynamic=False)
    management.soft_empty_cache()
    after = management.get_free_memory(device)
    logging.info("[LTX DFR decode memory] released %d other model(s); free budget %.0f -> %.0f MiB; VAE retained",
                 len(unloaded), before / 1024**2, after / 1024**2)
