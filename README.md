# ComfyUI-LTX25-DFR

**Lightricks' LTX-2.5 Diffusion Fidelity Rendering pipeline, ported to ComfyUI — with spatial refinement, temporal upscaling, and demonstrated runs on a 12 GB GPU.**

This project brings the LTX-2.5 DFR pipeline into a ComfyUI workflow: generate video and audio, refine spatial detail, increase the frame rate, and decode the result using DFR's generated keyframes. The stages are exposed as separate nodes, so you can stop after Stage 1 or build a workflow with spatial and temporal refinement.

This is an independent community project. It is not affiliated with, endorsed by, or maintained by Lightricks. The original models, DFR pipeline, and underlying research are the work of [Lightricks](https://github.com/Lightricks/LTX-2).

## What this port brings to ComfyUI

- **Spatial DFR:** Stage 1 generation, learned 2x latent spatial upscaling, and Stage 2 refinement with the detailing IC-LoRA.
- **Temporal DFR:** learned 2x temporal upscaling followed by diffusion refinement, with a second round available for approximately 4x temporal frame density / FPS. Temporal processing can start from either spatial stage.
- **Generated keyframes throughout the pipeline:** generated slots are carried between stages and used by the DFR decoder, alongside user image/keyframe conditioning.
- **Joint video and audio generation:** coordinated sampling state, separate output decoding, and audio trimming to match the final video duration.
- **DFR-aware tiled decoding:** spatial and temporal tiling, automatic tile scheduling, and a CUDA Triton attention path when available, with a PyTorch fallback.
- **ComfyUI integration:** works with ComfyUI model loaders and conditioning, with complete stage nodes for normal workflows while diagnostic and development implementations remain available in the source code.
- **Optional spatial refinement after temporal upscaling:** an additional route for combining the two types of refinement.

The port preserves upstream DFR behavior in areas such as conditioning, generated-keyframe layout, noise schedules, sampling-state handoffs, and decoding. It adapts execution to ComfyUI's model runtime and memory management. Identical pixels across different runtimes, precision settings, or hardware are not guaranteed.

## Performance on a 4070 Ti 12 GB GPU

The following timings were required to complete a full run under Windows using Comfy Desktop with SageAttention.

| Stages | Width | Height | MP* | Frames | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1S1T | 832 | 448 | 0.355469 | 193 | 84.32 |
| 1S | 1664 | 896 | 1.421875 | 97 | 164.65 |
| 2S1T | 1664 | 896 | 1.421875 | 193 | 355 |
| 2S | 1664 | 896 | 1.421875 | 97 | 156 |

Stage notation:

- **1S:** Spatial Stage 1 only.
- **2S:** Spatial Stage 1 followed by Spatial Stage 2.
- **1T:** one 2x temporal-upscaling round after the selected spatial stages.
- **1S1T / 2S1T:** the corresponding spatial path followed by one temporal round.

Widths, heights, and frame counts describe the reported outputs. These are different workflow configurations, so the rows should not be read as a controlled comparison of the cost of adding a stage. The precise timing scope has not yet been standardized. Performance depends on model precision, offloading, resolution, frame count, and decode settings; 12 GB is a demonstrated configuration, not a guarantee for every workflow.

*MP preserves the benchmark's calculation: width × height / 1,048,576.*

## Installation

1. Place this repository in your active ComfyUI installation's `custom_nodes` directory:

   ```text
   ComfyUI/
   └── custom_nodes/
       └── ComfyUI-LTX25-DFR/
           ├── __init__.py
           ├── nodes.py
           ├── DFR_Spatial/
           ├── DFR_Temporal/
           └── Workflow/
   ```

2. Use a ComfyUI build with LTX-2.5 support and install the model assets below.
3. Restart ComfyUI.
4. Load [Workflow/00_BaseTemplate.json](Workflow/00_BaseTemplate.json).

The node pack uses ComfyUI's Python environment and its existing PyTorch and safetensors dependencies. Triton acceleration is optional; the decoder has a PyTorch fallback when it is unavailable (extremely slow and less tested).

The bundled workflow also uses helper nodes from **ComfyUI-KJNodes**, **rgthree-comfy**, and **ComfyMath**. Install missing helper packs through ComfyUI's missing-node tools, or replace those helpers with equivalent wiring and primitive nodes. They are workflow conveniences rather than the DFR implementation itself.

## Models

Use the **LTX-2.5 distilled transformer** for this pipeline. The bundled template selects the ComfyUI INT8 ConvRot transformer and text encoder. Model weights are downloaded separately.

| Asset | ComfyUI model folder | Used for |
| --- | --- | --- |
| LTX-2.5 distilled transformer | `models/diffusion_models/` | Video/audio generation and refinement |
| Compatible LTX-2.5 text encoder | `models/text_encoders/` | Prompt encoding |
| LTX-2.5 video VAE | `models/vae/` | Image conditioning and video decoding |
| LTX-2.5 audio VAE | `models/vae/` | Audio decoding |
| LTX-2.5 latent spatial upscaler x2 | `models/latent_upscale_models/` | Spatial Stage 2 / post-temporal spatial upscaling |
| LTX-2.5 latent temporal upscaler x2 | `models/latent_upscale_models/` | Temporal refinement |
| LTX-2.5 pixel spatial upscaler detailing IC-LoRA | `models/loras/` | Spatial detailing |

Get the base assets from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) and the detailing adapter from [Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler).

Select your installed files in every loader, including the DFR video decoder's checkpoint selector. The DFR decoder needs access to the compatible video VAE checkpoint's keyframe weights.

## Using the workflow

Start with the included template and replace its example images and prompts with your own. Check the model selectors, dimensions, frame count, and FPS before running it.

1. **Set up the canvas and conditioning.** Resolve the DFR working canvas, encode the prompt, and supply any image/keyframe conditioning. The canvas resolver handles internal frame padding; retain its layout for downstream trimming.
2. **Run Spatial Stage 1.** This generates the initial video/audio state and DFR keyframes. Use the Stage 1 output helpers when stopping here.
3. **Optionally run Spatial Stage 2.** Upscale the latent, prepare the detailing model and conditioning, then run the refinement loop.
4. **Optionally add Temporal DFR.** Prepare the handoff from Stage 1 or Stage 2 and run a temporal round. One round maps 97 frames to 193 using `2 × (frames − 1) + 1`, with a corresponding increase in FPS.
5. **Decode and save.** Use the output helpers and DFR decoder for the selected path, decode and trim the audio, then connect the result to ComfyUI's video creation/save nodes. For temporal output, use the resulting FPS when creating the video.

The main nodes are grouped under **LTX / DFR Spatial** and **LTX / DFR Temporal**. Development, validation, and experimental nodes are retained in the source code but are not registered in the normal ComfyUI node menu.

## Practical notes

- **Memory:** begin with the template's quantized models and tiled decoding. If you run out of VRAM, reduce resolution, frame count, or decoder tile sizes. Offloading can also require substantial system RAM.
- **Model compatibility:** the DFR Model Capability Preflight checks the model's generated-keyframe support. Resolve a failed preflight before sampling.
- **Conditioning:** keep the DFR handoffs connected between stages. They carry information beyond the plain video latent, including generated keyframes and sampling state.
- **Final decoding:** a generic VAE decode does not consume the additional DFR handoff data. Use the provided decoder for the full DFR output path.
- **Experiments:** dense generated slots and the extra Stage 2 step are explicitly experimental variations of the upstream procedure.

When reporting an issue, include your ComfyUI version, GPU/VRAM, selected models and precision, workflow JSON, output dimensions/frame count, and the relevant console traceback.

## Credits and license

The original LTX-2.5 models, Diffusion Fidelity Rendering pipeline, and ported algorithms are by **Lightricks Ltd.** This repository adapts that work into ComfyUI nodes, stage handoffs, and decoding workflows. 

This port is distributed under the **LTX-2.x Community License Agreement**, dated August 11, 2026. See [LICENSE](LICENSE) for the complete agreement, including its attachments and use restrictions, and the [upstream license](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x).

This is a custom community license with use and commercial conditions, not an MIT or Apache license. Model downloads and third-party dependencies remain subject to their applicable licenses. Redistribution must preserve the applicable attribution notices and identify modifications to upstream files.
