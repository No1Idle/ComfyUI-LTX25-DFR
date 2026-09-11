"""Experimental one-block, spatially streamed Stage-5 executor.

Not registered as a node. Full temporal/video and keyframe states live on CPU;
each output core is written exactly once. Comparison target is a globally
initialized untiled block, NOT the production decoder's independently noised tiles.
"""
from collections import defaultdict
from typing import Any
import time
import torch
from .decoder_det_dual_stream_exact import _rope_matrices_for_times
from .decoder_joint_attention import joint_na3d
from .decoder_stage5_chunked_exact import inject_deferred_context_exact, inject_deferred_keyframe_context_exact
from .decoder_stage5_swiglu_chunked import residual_modulating_mlp_exact


def spatial_cores(height, width, core_height, core_width, halo, alignment=2):
    if min(core_height, core_width) < 1:
        raise ValueError('Core dimensions must be positive.')
    for y in range(0, height, core_height):
        for x in range(0, width, core_width):
            y1, x1 = min(height,y+core_height), min(width,x+core_width)
            lo_y=max(0,((y-halo)//alignment)*alignment)
            lo_x=max(0,((x-halo)//alignment)*alignment)
            hi_y=min(height,((y1+halo+alignment-1)//alignment)*alignment)
            hi_x=min(width,((x1+halo+alignment-1)//alignment)*alignment)
            yield (y,y1,x,x1),(lo_y,hi_y,lo_x,hi_x)


class BlockTimer:
    def __init__(self,device):
        self.device=device
        self.totals=defaultdict(float)
    def run(self,label,fn):
        torch.cuda.synchronize(self.device)
        start=time.perf_counter()
        result=fn()
        torch.cuda.synchronize(self.device)
        self.totals[label]+=time.perf_counter()-start
        return result


def block_slab(na,kitchen,block,upsample,video,keys,context,key_context,modulation,times,valid,origin,core,timer,drop_leading_frame=True):
    """Compute context+joint attention on halo input, MLP on owned cores only."""
    def context_injection():
        inject_deferred_context_exact(video,context,upsample,block.context_proj,w_chunks=4,drop_leading_frame=drop_leading_frame)
        inject_deferred_keyframe_context_exact(keys,key_context,upsample,block.context_proj,w_chunks=4)
    timer.run('context',context_injection)
    chunks=[modulation[i]+block.scale_shift_table[i].view(1,1,1,1,-1) for i in range(7)]
    scale,shift,_,mlp_scale,mlp_shift,_,_=chunks
    def attention():
        vnorm=block.norm1(video)*(1+scale)+shift
        knorm=block.norm1(keys)*(1+scale)+shift
        ypos=torch.arange(video.shape[2],device=video.device,dtype=torch.float32)+origin[0]
        xpos=torch.arange(video.shape[3],device=video.device,dtype=torch.float32)+origin[1]
        q,k,v=_qkv(na,kitchen,block.attn,vnorm,torch.arange(video.shape[1],device=video.device,dtype=torch.float32),height_positions=ypos,width_positions=xpos,separate_qkv=True)
        del vnorm
        kq,kk,kv=_qkv(na,kitchen,block.attn,knorm,times,height_positions=ypos,width_positions=xpos,separate_qkv=True)
        del knorm
        av,ak=joint_na3d(q,k,v,kq,kk,kv,times,valid,tuple(block.attn.kernel_size),num_slots=2)
        del q,k,v,kq,kk,kv
        sy,sx=core
        # Projection/MLP are pointwise: compute only the owned core.
        def projected(base,out):
            out=out[:,:,sy,sx].reshape(*base[:,:,sy,sx].shape).contiguous()
            dest=base[:,:,sy,sx].contiguous()
            chunk=max(1,2**25//max(dest.shape[2]*dest.shape[3]*dest.shape[4],1))
            for t in range(0,dest.shape[1],chunk):dest[:,t:t+chunk].add_(block.attn.proj(out[:,t:t+chunk]))
            return dest
        return projected(video,av),projected(keys,ak)
    vout,kout=timer.run('attention_and_projection',attention)
    def mlp():
        vo=residual_modulating_mlp_exact(vout,block.mlp,block.norm2,mlp_scale,mlp_shift)
        ko=residual_modulating_mlp_exact(kout,block.mlp,block.norm2,mlp_scale,mlp_shift)
        ko.mul_(valid.to(ko.dtype)[None,:,None,None,None])
        return vo,ko
    return timer.run('feed_forward',mlp)


@torch.inference_mode()
def stream_one_block(na,kitchen,block,upsample,video,keys,context,key_context,modulation,times,valid,*,core_height=96,core_width=144):
    tensors=(video,keys,context,key_context)
    if any(t.device.type!='cpu' for t in tensors):raise ValueError('Full states must reside on CPU.')
    if any(t.ndim != 5 for t in tensors):raise ValueError('Expected channels-last B,T/P,H,W,C tensors.')
    if any(t.dtype != video.dtype for t in tensors):raise ValueError('All CPU states must have the same dtype.')
    if video.shape[0]!=1 or video.shape[2:4]!=keys.shape[2:4]:raise ValueError('Prototype requires batch 1 and matching spatial geometry.')
    if tuple(upsample.stride)!=(2,2,2):raise ValueError('Prototype requires Stage-4 stride (2,2,2).')
    if video.shape[2]!=2*context.shape[2] or video.shape[3]!=2*context.shape[3] or video.shape[1]!=2*context.shape[1]-1:raise ValueError('Context/video geometry mismatch.')
    if keys.shape[1] != key_context.shape[1] or keys.shape[2] != 2*key_context.shape[2] or keys.shape[3] != 2*key_context.shape[3]:raise ValueError('Keyframe context geometry mismatch.')
    if times.shape != (keys.shape[1],) or valid.shape != times.shape:raise ValueError('Keyframe times/valid geometry mismatch.')
    device=next(block.parameters()).device
    if device.type != 'cuda':raise ValueError('Prototype execution requires CUDA.')
    timer=BlockTimer(device);started=time.perf_counter()
    outputs=(torch.empty_like(video),torch.empty_like(keys))
    halo=max(int(block.attn.kernel_size[1])//2,int(block.attn.kernel_size[2])//2)
    count=processed=0;upload_bytes=download_bytes=0
    torch.cuda.reset_peak_memory_stats(device)
    for owned,extended in spatial_cores(video.shape[2],video.shape[3],core_height,core_width,halo):
        y,y1,x,x1=owned;ly,hy,lx,hx=extended
        slices=(video[:,:,ly:hy,lx:hx],keys[:,:,ly:hy,lx:hx],context[:,:,ly//2:hy//2,lx//2:hx//2],key_context[:,:,ly//2:hy//2,lx//2:hx//2])
        upload_bytes+=sum(t.numel()*t.element_size() for t in slices)
        def upload():
            # Bounded pinned staging, not full-video pinned allocations.
            return tuple(t.contiguous().pin_memory().to(device,non_blocking=False) for t in slices)
        v,k,c,kc=timer.run('pack_and_upload',upload)
        vo,ko=block_slab(na,kitchen,block,upsample,v,k,c,kc,modulation,times,valid,(ly,lx),(slice(y-ly,y1-ly),slice(x-lx,x1-lx)),timer)
        download_bytes+=vo.numel()*vo.element_size()+ko.numel()*ko.element_size()
        def download():
            outputs[0][:,:,y:y1,x:x1].copy_(vo.cpu())
            outputs[1][:,:,y:y1,x:x1].copy_(ko.cpu())
        timer.run('download_and_scatter',download)
        del v,k,c,kc,vo,ko
        # Keep the standalone probe below Windows' shared-memory cliff.
        timer.run('release_cache', torch.cuda.empty_cache)
        count+=1;processed+=(hy-ly)*(hx-lx)
    report=dict(wall_seconds=time.perf_counter()-started,phases=dict(timer.totals),cores=count,spatial_coverage=processed/(video.shape[2]*video.shape[3]),upload_GiB=upload_bytes/2**30,download_GiB=download_bytes/2**30,peak_cuda_MiB=torch.cuda.max_memory_allocated(device)/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/2**20)
    return outputs,report


def _qkv(
    comfy_na: Any,
    comfy_kitchen: Any,
    attn: Any,
    x: torch.Tensor,
    temporal_positions: torch.Tensor,
    *,
    width_positions: torch.Tensor | None = None,
    height_positions: torch.Tensor | None = None,
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
            height_positions=height_positions,
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

class AsyncBlockTimer:
    """CUDA events only: no phase-level synchronization that would defeat overlap."""
    def __init__(self, device):
        self.device = device
        self.events = []
    def run(self, label, fn):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        stream = torch.cuda.current_stream(self.device)
        start.record(stream)
        result = fn()
        end.record(stream)
        self.events.append((label, start, end))
        return result
    def totals(self):
        totals = defaultdict(float)
        for label, start, end in self.events:
            totals[label] += start.elapsed_time(end)/1000
        return dict(totals)


@torch.inference_mode()
def stream_one_block_overlap(na,kitchen,block,upsample,video,keys,context,key_context,modulation,times,valid,*,core_height=80,core_width=128,interrupt_check=None,drop_leading_frame=True):
    """Two-core pipeline with separate H2D/compute/D2H streams and bounded staging.

    Main thread packs the next core while queued GPU work proceeds. Pending
    downloads retain their source tensors and pinned destinations until complete.
    Event dependencies protect every cross-stream use; CPU scatter only follows
    completion of the matching download. Final drain is included in wall time.
    """
    tensors=(video,keys,context,key_context)
    if any(t.device.type!='cpu' or t.ndim!=5 for t in tensors):
        raise ValueError('Full states must be rank-5 CPU tensors.')
    if video.shape[0]!=1 or video.shape[2:4]!=keys.shape[2:4]:
        raise ValueError('Prototype requires batch one and matching spatial geometry.')
    if tuple(upsample.stride)!=(2,2,2) or tuple(video.shape[1:4])!=(2*context.shape[1]-int(drop_leading_frame),2*context.shape[2],2*context.shape[3]):
        raise ValueError('Video/context geometry mismatch.')
    if keys.shape[1]!=key_context.shape[1] or tuple(keys.shape[2:4])!=tuple(2*n for n in key_context.shape[2:4]):
        raise ValueError('Keyframe context geometry mismatch.')
    device=next(block.parameters()).device
    if device.type!='cuda':raise ValueError('CUDA required.')
    torch.cuda.synchronize(device)
    started=time.perf_counter()
    outputs=(torch.empty_like(video),torch.empty_like(keys))
    timer=AsyncBlockTimer(device)
    compute=torch.cuda.current_stream(device)
    upload_stream=torch.cuda.Stream(device=device)
    download_stream=torch.cuda.Stream(device=device)
    schedule=list(spatial_cores(video.shape[2],video.shape[3],core_height,core_width,max(block.attn.kernel_size[1]//2,block.attn.kernel_size[2]//2)))
    host_pack=host_scatter=0.;upload_bytes=download_bytes=0
    torch.cuda.reset_peak_memory_stats(device)

    def prepare(index):
        nonlocal host_pack,upload_bytes
        owned,extended=schedule[index];ly,hy,lx,hx=extended
        views=(video[:,:,ly:hy,lx:hx],keys[:,:,ly:hy,lx:hx],context[:,:,ly//2:hy//2,lx//2:hx//2],key_context[:,:,ly//2:hy//2,lx//2:hx//2])
        start=time.perf_counter()
        # One direct copy into pinned memory, avoiding an extra contiguous CPU copy.
        host=tuple(torch.empty(t.shape,dtype=t.dtype,device='cpu',pin_memory=True) for t in views)
        for dest,source in zip(host,views):dest.copy_(source)
        host_pack+=time.perf_counter()-start
        upload_bytes+=sum(t.numel()*t.element_size() for t in host)
        with torch.cuda.stream(upload_stream):
            gpu=timer.run('upload_gpu',lambda:tuple(t.to(device,non_blocking=True) for t in host))
            ready=torch.cuda.Event();ready.record(upload_stream)
        return owned,extended,host,gpu,ready

    def finish(pending):
        nonlocal host_scatter
        owned,host,gpu_outputs,item,done=pending
        done.synchronize()
        y,y1,x,x1=owned
        start=time.perf_counter()
        for dest,source in zip(outputs,host):dest[:,:,y:y1,x:x1].copy_(source)
        host_scatter+=time.perf_counter()-start

    current=None;pending=None
    cache_trims=0
    try:
        current=prepare(0)
        for index in range(len(schedule)):
            if interrupt_check is not None: interrupt_check()
            # Bound cached cross-stream allocations before they cause WDDM spill.
            if torch.cuda.memory_reserved(device) > 6 * 1024**3:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                cache_trims += 1
            owned,extended,host_inputs,gpu,ready=current
            y,y1,x,x1=owned;ly,hy,lx,hx=extended
            compute.wait_event(ready)
            for t in gpu:t.record_stream(compute)
            result=block_slab(na,kitchen,block,upsample,*gpu,modulation,times,valid,(ly,lx),(slice(y-ly,y1-ly),slice(x-lx,x1-lx)),timer,drop_leading_frame=drop_leading_frame)
            calculated=torch.cuda.Event();calculated.record(compute)
            host_outputs=tuple(torch.empty(t.shape,dtype=t.dtype,device='cpu',pin_memory=True) for t in result)
            download_bytes+=sum(t.numel()*t.element_size() for t in result)
            with torch.cuda.stream(download_stream):
                download_stream.wait_event(calculated)
                for t in result:t.record_stream(download_stream)
                def copy_outputs():
                    for dest,source in zip(host_outputs,result):dest.copy_(source,non_blocking=True)
                timer.run('download_gpu',copy_outputs)
                done=torch.cuda.Event();done.record(download_stream)
            # Current computation is already queued while the previous result drains.
            if pending is not None:finish(pending)
            pending=(owned,host_outputs,result,current,done)
            del host_inputs,gpu,result,host_outputs,current
            current=prepare(index+1) if index+1<len(schedule) else None
        if pending is not None:finish(pending)
        pending=None
        torch.cuda.synchronize(device)
        report=dict(wall_seconds=time.perf_counter()-started,gpu_phases=timer.totals(),host_pack_seconds=host_pack,host_scatter_seconds=host_scatter,cache_trims=cache_trims,cores=len(schedule),upload_GiB=upload_bytes/2**30,download_GiB=download_bytes/2**30,peak_cuda_MiB=torch.cuda.max_memory_allocated(device)/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/2**20)
        return outputs,report
    finally:
        # Keep all source/destination references alive until outstanding copies finish,
        # including exceptional exits. No background transfers outlive this call.
        torch.cuda.synchronize(device)
