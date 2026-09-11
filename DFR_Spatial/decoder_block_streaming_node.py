"""Experimental global-initialization decoder with CPU-backed Stage-5 blocks."""
import logging
import time
from dataclasses import replace
import torch
from . import decoder_pipeline as prep
from .decoder_block_streaming_prototype import spatial_cores, _qkv, stream_one_block_overlap
from .decoder_det_dual_stream_exact import _load_comfy_vae_for_execution
from .decoder_joint_attention import joint_na3d
from .decoder_bridge import restore_decoder_generator
from .decoder_official_tiling import crop_trailing_context_natten_pad
from .decoder_det_stage_official_preflight import keyframe_clip_times_official
from .decoder_stage5_chunked_exact import _pixels_from_stage5
from .decoder_full_tiled_exact import prepare_official_tiled_decode_schedule, prepare_official_auto_tiled_decode_schedule, planes_for_tile_official
from .decoder_official_tiling import slice_stage4_tile

log=logging.getLogger(__name__)


def _stage4_cpu(na,kitchen,decoder,execution,core_h,core_w,check):
    video,keys=execution.stage4_input_video,execution.stage4_input_keyframes
    device=next(decoder.parameters()).device
    times=execution.stage4_input_times.to(device)
    valid=execution.stage4_input_valid.to(device)
    outputs=torch.empty_like(video),torch.empty_like(keys)
    blocks=decoder.det_stages[3]
    halo=sum(max(b.attn.kernel_size[1]//2,b.attn.kernel_size[2]//2) for b in blocks)
    for owned,extended in spatial_cores(video.shape[2],video.shape[3],core_h,core_w,halo,alignment=1):
        check()
        y,y1,x,x1=owned;ly,hy,lx,hx=extended
        v=video[:,:,ly:hy,lx:hx].to(device).contiguous()
        k=keys[:,:,ly:hy,lx:hx].to(device).contiguous()
        ypos=torch.arange(hy-ly,device=device,dtype=torch.float32)+ly
        xpos=torch.arange(hx-lx,device=device,dtype=torch.float32)+lx
        tpos=torch.arange(video.shape[1],device=device,dtype=torch.float32)
        for block in blocks:
            q,kk,vv=_qkv(na,kitchen,block.attn,block.norm1(v),tpos,height_positions=ypos,width_positions=xpos)
            kq,kkk,kv=_qkv(na,kitchen,block.attn,block.norm1(k),times,height_positions=ypos,width_positions=xpos)
            av,ak=joint_na3d(q,kk,vv,kq,kkk,kv,times,valid,tuple(block.attn.kernel_size),num_slots=2)
            del q,kk,vv,kq,kkk,kv
            v=v+block.attn.proj(av.reshape(v.shape))
            k=k+block.attn.proj(ak.reshape(k.shape))
            del av,ak
            v=block.mlp(v,pre=block.norm2,add_to=v)
            k=block.mlp(k,pre=block.norm2,add_to=k)
        sy,sx=slice(y-ly,y1-ly),slice(x-lx,x1-lx)
        outputs[0][:,:,y:y1,x:x1].copy_(v[:,:,sy,sx].cpu())
        outputs[1][:,:,y:y1,x:x1].copy_(k[:,:,sy,sx].cpu())
        del v,k
        torch.cuda.empty_cache()
    return outputs


def _initialize_hidden(na,decoder,generator,frames,height,width,planes,valid,core_h,core_w,check):
    """Draw one global video then keyframe noise field; independent of core sizes."""
    device=next(decoder.parameters()).device;dtype=next(decoder.parameters()).dtype
    patch=int(decoder.patch_size);channels=int(decoder.stage_channels[-1]) if hasattr(decoder,'stage_channels') else int(decoder.diff_blocks[0].attn.dim)
    video_noise=torch.randn((1,decoder.out_channels,frames,height*patch,width*patch),device=generator.device,dtype=dtype,generator=generator).cpu()
    key_noise=torch.randn((1,decoder.out_channels,planes,height*patch,width*patch),device=generator.device,dtype=dtype,generator=generator).cpu()
    outputs=(torch.empty((1,frames,height,width,channels),dtype=dtype),torch.empty((1,planes,height,width,channels),dtype=dtype))
    for (y,y1,x,x1),_ in spatial_cores(height,width,core_h,core_w,0,alignment=1):
        check()
        for index,noise in enumerate((video_noise,key_noise)):
            pixels=noise[:,:,:,y*patch:y1*patch,x*patch:x1*patch].to(device)
            patched=na.patchify(pixels,patch_size_hw=patch,patch_size_t=1)
            hidden=decoder.conv_in_x_t(patched.permute(0,2,3,4,1)).contiguous()
            if index: hidden.mul_(valid.to(device=device,dtype=dtype)[None,:,None,None,None])
            outputs[index][:,:,y:y1,x:x1].copy_(hidden.cpu())
            del pixels,patched,hidden
        torch.cuda.empty_cache()
    return outputs


def _output_images(na,decoder,hidden,frames,pixel_h,pixel_w,core_h,core_w,output_dtype,check,normalize=True):
    device=next(decoder.parameters()).device;patch=int(decoder.patch_size)
    images=torch.empty((frames,pixel_h,pixel_w,decoder.out_channels),dtype=output_dtype,device='cpu')
    for (y,y1,x,x1),_ in spatial_cores(hidden.shape[2],hidden.shape[3],core_h,core_w,0,alignment=1):
        check()
        features=hidden[:,:frames,y:y1,x:x1].to(device).contiguous()
        pixels=_pixels_from_stage5(na,decoder,features)
        out=pixels[0].permute(1,2,3,0).contiguous()
        if normalize: out.add_(1).mul_(.5).clamp_(0,1)
        images[:,y*patch:min(y1*patch,pixel_h),x*patch:min(x1*patch,pixel_w)].copy_(out[:,:min((y1-y)*patch,pixel_h-y*patch),:min((x1-x)*patch,pixel_w-x*patch)].cpu())
        del features,pixels,out
        torch.cuda.empty_cache()
    return images


def _auto_streaming_core_sizes(decoder, config):
    """Fit owned cores plus both execution halos inside classic tile bounds.

    The classic planner remains a heuristic for this different executor. This
    conversion avoids treating overlapping tile dimensions as owned core sizes.
    """
    patch = int(decoder.patch_size)
    alignment = 2  # Stage-5 spatial_cores aligns halo slices to two tokens.
    halo5 = max(max(int(b.attn.kernel_size[1])//2, int(b.attn.kernel_size[2])//2)
                for b in decoder.diff_blocks)
    halo5_pixels = ((halo5 + alignment - 1)//alignment)*alignment*patch
    halo4 = sum(max(int(b.attn.kernel_size[1])//2, int(b.attn.kernel_size[2])//2)
                for b in decoder.det_stages[3])
    halo4_pixels = halo4 * patch * max(int(x) for x in decoder.upsamples[3].stride[1:])
    border = max(halo4_pixels, halo5_pixels)
    cores = tuple(((int(size)-2*border)//8)*8
                  for size in (config.height.tile_size, config.width.tile_size))
    if min(cores) < 32:
        raise ValueError('Automatic tile is too small for experimental streaming halos; increase auto_tile_multiplier or use manual cores.')
    return cores


@torch.inference_mode()
def decode_experimental(stage_2_handoff,final_video_latent,dfr_layout,vae,vae_name,*,core_height=320,core_width=512,tile_frames=104,use_auto_tiling=False,auto_tile_multiplier=1.0):
    if not use_auto_tiling and (core_height<32 or core_width<32 or core_height%8 or core_width%8):
        raise ValueError('Experimental core dimensions must be multiples of 8 and at least 32 pixels.')
    if torch.device(vae.device).type!='cuda':raise ValueError('Experimental streamed decode requires CUDA.')
    if not use_auto_tiling and (tile_frames < 80 or tile_frames % 8):
        raise ValueError('tile_frames must be a multiple of 8 and at least 80.')
    started=time.perf_counter()
    latent=prep._validated_final_video_latent(stage_2_handoff,final_video_latent,dfr_layout)
    handoff,keyframes=prep.prepare_official_final_decode(stage_2_handoff,dfr_layout)
    prep.prepare_decoder_memory(vae)
    substrate,_=prep.build_decoder_keyframe_substrate(vae,handoff,keyframes,dfr_layout,collect_diagnostics=False)
    weights,_=prep.extract_decoder_keyframe_checkpoint_weights(substrate,str(vae_name))
    joint,_=prep.prepare_joint_decoder_attention(substrate,weights)
    context,_=prep.prepare_deterministic_decoder_keyframe_context(vae,substrate,weights,joint,keyframes,dfr_layout)
    preflight,_=prep.prepare_official_deterministic_stage_preflight(vae,context,collect_diagnostics=False)
    port,_=prep.prepare_exact_deterministic_dual_stream_port(preflight,joint)
    execution,_=prep.execute_exact_deterministic_stages(vae,latent,context,port,collect_diagnostics=False)
    mm,na,kitchen=_load_comfy_vae_for_execution(vae,torch.empty(execution.input_video_shape,device='meta',dtype=vae.vae_dtype))
    decoder=vae.first_stage_model.decoder
    if tuple(decoder.upsamples[3].stride)!=(2,2,2) or int(decoder.patch_size)!=4 or len(decoder.diff_blocks)!=8:
        raise ValueError('Experimental decoder requires patch size 4, final stride (2,2,2), and eight Stage-5 blocks.')
    config=getattr(vae.first_stage_model,'config',{}) or {}
    if str(config.get('model_output_type',getattr(decoder,'model_output_type','')))!='x0' or decoder.default_inference_timesteps.numel()!=1:
        raise ValueError('Experimental decoder requires one-step x0 DiffVAE.')
    check=mm.throw_exception_if_processing_interrupted
    execution=replace(execution,stage4_input_video=execution.stage4_input_video.cpu(),stage4_input_keyframes=execution.stage4_input_keyframes.cpu())
    mm.soft_empty_cache()
    remaining=tuple(int(v) for v in context.temporal_scale_schedule)
    content_frames=(execution.input_video_shape[2]-1)*remaining[0]+1
    pixel_h,pixel_w=(int(n) for n in context.final_pixel_spatial_shape)
    if use_auto_tiling:
        auto_schedule,_=prepare_official_auto_tiled_decode_schedule(
            vae,execution,context,auto_tile_multiplier=auto_tile_multiplier)
        auto_config=auto_schedule.tiling_config
        core_height,core_width=_auto_streaming_core_sizes(decoder,auto_config)
        tile_frames=int(auto_config.frames.tile_size)
        log.info('[LTX DFR experimental autotile] classic tile=%dx%dx%d -> frames=%d; owned cores=%dx%d; multiplier=%.2f',
                 auto_config.frames.tile_size,auto_config.height.tile_size,auto_config.width.tile_size,
                 tile_frames,core_height,core_width,auto_tile_multiplier)
        del auto_schedule
    ch,cw=core_height//4,core_width//4
    generator,_=restore_decoder_generator(handoff)
    output_dtype=vae.vae_output_dtype() if callable(getattr(vae,'vae_output_dtype',None)) else mm.intermediate_dtype()
    indices=torch.tensor(context.keyframe_pixel_frame_indices,dtype=torch.long)
    # Keep the previous single-window initialization/keyframe set for short clips.
    if content_frames <= tile_frames:
        windows=[(execution,indices,0,int(content_frames),True,True,torch.ones(1))]
        overlap=0
    else:
        overlap,spatial_overlap=prep._required_manual_overlaps(vae)
        schedule,_=prepare_official_tiled_decode_schedule(
            vae,execution,context,tile_frames=tile_frames,temporal_overlap=overlap,
            tile_height=max(pixel_h,2*spatial_overlap),tile_width=max(pixel_w,2*spatial_overlap),spatial_overlap=spatial_overlap,
        )
        windows=[]
        for tile in schedule.tiles:
            # Full spatial extent: only the original temporal splitter is used.
            if tile.out_coords[3].indices(pixel_h)[:2] != (0,pixel_h) or tile.out_coords[4].indices(pixel_w)[:2] != (0,pixel_w):
                raise RuntimeError('Expected exactly one spatial extent per temporal window.')
            lo,hi,_=tile.out_coords[2].indices(int(content_frames))
            video,is_origin,pad_trailing,_=slice_stage4_tile(execution.stage4_input_video,tile,content_frames=schedule.content_stage4_shape[0])
            keep=planes_for_tile_official(indices,lo,hi-1,clip_start_frame=0)
            selected=indices[keep]
            stage4_origin=tile.in_coords[1].indices(schedule.content_stage4_shape[0])[0]
            window=replace(execution,
                stage4_input_video=video,
                stage4_input_keyframes=execution.stage4_input_keyframes[:,keep],
                stage4_input_times=keyframe_clip_times_official(selected,remaining[3],0,extra_origin=float(stage4_origin)),
                stage4_input_valid=execution.stage4_input_valid.cpu()[keep],
            )
            mask=tile.masks_1d[2].detach().float().cpu()[:hi-lo]
            windows.append((window,selected,lo,hi,is_origin,pad_trailing,mask))
    log.info('[LTX DFR experimental decode] temporal windows=%d; tile_frames=%d; overlap=%d; spatial cores=%dx%d',len(windows),tile_frames,overlap,core_height,core_width)
    images=None
    weights=torch.zeros(int(content_frames),dtype=torch.float32)
    block_seconds=0.
    for wi,(window,selected,lo,hi,is_origin,pad_trailing,mask) in enumerate(windows):
        check();mm.soft_empty_cache()
        log.info('[LTX DFR experimental decode] window %d/%d: frames [%d,%d); keyframes=%d',wi+1,len(windows),lo,hi,len(selected))
        c,kc=_stage4_cpu(na,kitchen,decoder,window,max(1,ch//2),max(1,cw//2),check)
        valid=window.stage4_input_valid.to(vae.device)
        if pad_trailing:
            c=crop_trailing_context_natten_pad(c,n_latent_frames=window.ghost_pad_latent_frames,time_scale=remaining[0]//2,stage5_kernel_t=max(1,-(-int(decoder.diff_blocks[0].attn.kernel_size[0])//2))).contiguous()
        kc.mul_(valid.cpu().to(kc.dtype)[None,:,None,None,None])
        canvas_frames=c.shape[1]*2-int(is_origin)
        v,k=_initialize_hidden(na,decoder,generator,canvas_frames,c.shape[2]*2,c.shape[3]*2,kc.shape[1],valid,ch,cw,check)
        times=keyframe_clip_times_official(selected,remaining[4],0,extra_origin=float(lo)).to(vae.device)
        timestep=decoder.default_inference_timesteps.to(device=vae.device,dtype=torch.float32)[0].expand(1)
        modulation=decoder.shared_adaln(decoder.t_embedder(float(decoder.timestep_scale_multiplier)*timestep,dtype=v.dtype))
        for index,block in enumerate(decoder.diff_blocks):
            check();mm.soft_empty_cache()
            (v,k),report=stream_one_block_overlap(na,kitchen,block,decoder.upsamples[3],v,k,c,kc,modulation,times,valid,core_height=ch,core_width=cw,interrupt_check=check,drop_leading_frame=is_origin)
            block_seconds+=report['wall_seconds']
            log.info('[LTX DFR experimental decode] window %d/%d block %d/8: %.2fs; peak_reserved=%.0f MiB',wi+1,len(windows),index+1,report['wall_seconds'],report['peak_reserved_MiB'])
        del k,c,kc
        part=_output_images(na,decoder,v,hi-lo,pixel_h,pixel_w,ch,cw,output_dtype,check,normalize=len(windows)==1)
        del v
        if len(windows)==1:
            images=part
        else:
            if images is None: images=torch.zeros((int(content_frames),pixel_h,pixel_w,decoder.out_channels),dtype=output_dtype)
            # Blend with the original temporal masks. Frame-wise accumulation bounds scratch RAM.
            for local in range(hi-lo):
                strength=float(mask[local] if mask.numel()>1 else mask[0])
                images[lo+local].add_(part[local],alpha=strength)
                weights[lo+local]+=strength
            del part
    if len(windows)>1:
        if not bool((weights>0).all()):raise RuntimeError('Temporal windows left uncovered output frames.')
        for frame in range(int(content_frames)): images[frame].div_(weights[frame]).add_(1).mul_(.5).clamp_(0,1)
    report=f'Experimental temporal-window decode: {time.perf_counter()-started:.2f}s total; tiling={'auto' if use_auto_tiling else 'manual'}; cores={core_height}x{core_width}; windows={len(windows)}, tile_frames={tile_frames}, overlap={overlap}; Stage 5 blocks {block_seconds:.2f}s; output={tuple(images.shape)}. Noise is global within each temporal window.'
    log.info('[LTX DFR experimental decode] %s',report)
    return images,report
