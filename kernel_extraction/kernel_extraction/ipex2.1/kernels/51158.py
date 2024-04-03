

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/7o/c7ocdny7fh6tv2mnaue4apq6bxjt2mlusuxt6d22yt6ytprsruqk.py
# Source Nodes: [setitem_9], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_9 => copy_9, slice_scatter_33, slice_scatter_34
triton_poi_fused_copy_slice_scatter_12 = async_compile.triton('triton_poi_fused_copy_slice_scatter_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp47 = tl.load(in_ptr2 + (y0 + (4096*x1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to(x1, [XBLOCK, YBLOCK])
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1, 1], 256, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = (((-256) + x1 + (513*y0)) // 512) % 513
    tmp10 = tl.full([1, 1], 512, tl.int64)
    tmp11 = tmp9 < tmp10
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + ((256*((((-256) + x1 + (513*y0)) // 262656) % 15)) + ((((-256) + x1 + (513*y0)) // 512) % 513)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + ((256*((((-256) + x1 + (513*y0)) // 262656) % 15)) + (((-256) + x1 + (513*y0)) % 512)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.where(tmp12, tmp15, 0.0)
    tmp17 = tl.where(tmp8, tmp16, 0.0)
    tmp18 = tl.full([1, 1], 0, tl.int64)
    tmp19 = tmp18 >= tmp1
    tmp20 = tmp19 & tmp2
    tmp21 = tmp6 & tmp20
    tmp22 = (((-131584) + x1 + (513*y0)) // 512) % 513
    tmp23 = tmp22 < tmp10
    tmp24 = tmp23 & tmp21
    tmp25 = tl.load(in_ptr0 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 15)) + ((((-131584) + x1 + (513*y0)) // 512) % 513)), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 15)) + ((x1 + (513*y0)) % 512)), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.where(tmp24, tmp27, 0.0)
    tmp29 = tl.where(tmp21, tmp28, 0.0)
    tmp30 = tl.load(in_ptr2 + (y0 + (4096*x1)), tmp20 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp6, tmp29, tmp30)
    tmp32 = tl.where(tmp20, tmp31, 0.0)
    tmp33 = tl.load(in_ptr2 + (y0 + (4096*x1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tl.where(tmp19, tmp32, tmp33)
    tmp35 = tl.where(tmp7, tmp17, tmp34)
    tmp36 = tl.where(tmp2, tmp35, 0.0)
    tmp37 = tmp6 & tmp19
    tmp38 = tmp23 & tmp37
    tmp39 = tl.load(in_ptr0 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 15)) + ((((-131584) + x1 + (513*y0)) // 512) % 513)), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr1 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 15)) + ((x1 + (513*y0)) % 512)), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.where(tmp38, tmp41, 0.0)
    tmp43 = tl.where(tmp37, tmp42, 0.0)
    tmp44 = tl.load(in_ptr2 + (y0 + (4096*x1)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp45 = tl.where(tmp6, tmp43, tmp44)
    tmp46 = tl.where(tmp19, tmp45, 0.0)
    tmp48 = tl.where(tmp19, tmp46, tmp47)
    tmp49 = tl.where(tmp2, tmp36, tmp48)
    tl.store(out_ptr0 + (x1 + (513*y0)), tmp49, xmask & ymask)
''')
