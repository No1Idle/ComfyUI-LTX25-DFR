# ComfyUI-LTX25-DFR

**Lightricks' LTX-2.5 Diffusion Fidelity Rendering pipeline, ported to ComfyUI — with spatial refinement, temporal upscaling, and demonstrated runs on a 12 GB GPU.**

This project brings the LTX-2.5 DFR pipeline into a ComfyUI workflow: generate video and audio, refine spatial detail, increase the frame rate, and decode the result using DFR's generated keyframes. The stages are exposed as separate nodes, so you can stop after Stage 1 or build a workflow with spatial and temporal refinement. The nodes are built to be reasonably modular.

This is an independent community project. It is not affiliated with, endorsed by, or maintained by Lightricks. The original models, DFR pipeline, and underlying research are the work of [Lightricks](https://github.com/Lightricks/LTX-2).

The DFR pipeline increases output quality and reduces the "smearing problem" associated with fast movements, even if in the current implementation it does not completely remove it.

Another focus of the project is to make generation as convenient as possible on mid-range consumer GPUs, around 12 GB of VRAM.

## What this port brings to ComfyUI

- **Spatial DFR:** Stage 1 generation, learned 2x latent spatial upscaling, and Stage 2 refinement with the detailing IC-LoRA.
- **Temporal DFR:** learned 2x temporal upscaling followed by diffusion refinement, with a second round available for approximately 4x temporal frame density / FPS. Temporal processing can start from either spatial stage.
- **DFR-aware tiled decoding:** spatial and temporal tiling, automatic tile scheduling, and a CUDA Triton attention path when available, with a PyTorch fallback.
- **ComfyUI integration:** works with ComfyUI model loaders and conditioning, with complete stage nodes for normal workflows, while diagnostic and development implementations remain available in the source code.
- **Modular combination of spatial and temporal upscaling:** the current version of the project allows spatial and temporal upscaling stages of the DFR pipeline to be combined in different orders.

## Performance on a 4070 Ti 12 GB GPU

The following timings were required to complete a full run under Windows using ComfyUI Desktop with SageAttention.

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

*The current table timings are outdated. The latest project version includes improved memory management and should be faster.*

## Installation

1. Place this repository in your active ComfyUI installation's `custom_nodes` directory:

   ```text
   ComfyUI/
   └── custom_nodes/
       └── ComfyUI-LTX25-DFR/
   ```

2. Use a ComfyUI build with LTX-2.5 support and install the model assets below.
3. Example workflows are included to help you get started with the project.

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

## Practical notes

- **Memory:** Both the decode nodes and the iteration nodes have options for memory management. The default options work on a 12 GB GPU, but they are not guaranteed to be optimal for every GPU. Changing the default values can significantly affect execution time.
- **Decoders:** There are two different decoders: a "classic" one that follows the original pipeline and an experimental one. Both decoders should produce equivalent results. The experimental decoder can be faster but requires more system RAM.
- **Auto Tiling:** The current auto-tiling system is experimental. It can be used as a good starting point, but in most cases ad-hoc manual settings can outperform it.

## Credits and license

The original LTX-2.5 models, Diffusion Fidelity Rendering pipeline, and ported algorithms are by **Lightricks Ltd.** This repository adapts that work into ComfyUI nodes, stage handoffs, and decoding workflows.

This port is distributed under the **LTX-2.x Community License Agreement**, dated August 11, 2026. See [LICENSE](LICENSE) for the complete agreement, including its attachments and use restrictions, and the [upstream license](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x).

This is a custom community license with use and commercial conditions, not an MIT or Apache license. Model downloads and third-party dependencies remain subject to their applicable licenses. Redistribution must preserve the applicable attribution notices and identify modifications to upstream files.