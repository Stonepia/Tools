

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/ba/cba4daq52ufq7ort5fkmmh6nyx3zb7xgvngknjiefq7xfs2g6mkq.py
# Source Nodes: [setitem_9], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_9 => copy_9, slice_scatter_33
triton_poi_fused_copy_slice_scatter_16 = async_compile.triton('triton_poi_fused_copy_slice_scatter_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1020
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 255
    y1 = (yindex // 255)
    y3 = yindex
    tmp22 = tl.load(in_ptr3 + (513 + x2 + (513*y0) + (131328*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = ((257 + x2 + (513*y0)) // 512)
    tmp7 = tl.full([1, 1], 512, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + ((256*((257 + x2 + (513*y0)) // 262656)) + (1024*y1) + (1024*((257 + x2 + (513*y0)) // 787968)) + ((257 + x2 + (513*y0)) // 512)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr1 + ((256*((257 + x2 + (513*y0)) // 262656)) + (1024*y1) + (1024*((257 + x2 + (513*y0)) // 787968)) + ((257 + x2 + (513*y0)) % 512)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tl.where(tmp9, tmp12, 0.0)
    tmp14 = tl.where(tmp5, tmp13, 0.0)
    tmp15 = tl.full([1, 1], 0, tl.int64)
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.load(in_ptr2 + ((-130815) + x2 + (513*y0) + (393984*y1)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.where(tmp16, tmp17, 0.0)
    tmp19 = tl.full([1, 1], 0, tl.int32)
    tmp20 = tl.full([1, 1], 3, tl.int32)
    tmp21 = tmp19 == tmp20
    tmp23 = tl.full([1, 1], 3, tl.int64)
    tmp24 = tmp15 < tmp23
    tmp25 = tl.broadcast_to(x2, [XBLOCK, YBLOCK])
    tmp26 = tmp25 >= tmp3
    tmp27 = tmp26 & tmp24
    tmp28 = tmp8 & tmp27
    tmp29 = tl.load(in_ptr0 + ((256*((257 + x2 + (513*y0)) // 262656)) + (1024*y1) + (1024*((257 + x2 + (513*y0)) // 787968)) + ((257 + x2 + (513*y0)) // 512)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr1 + ((256*((257 + x2 + (513*y0)) // 262656)) + (1024*y1) + (1024*((257 + x2 + (513*y0)) // 787968)) + ((257 + x2 + (513*y0)) % 512)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tl.where(tmp28, tmp31, 0.0)
    tmp33 = tl.where(tmp27, tmp32, 0.0)
    tmp34 = 0.0
    tmp35 = tl.where(tmp26, tmp33, tmp34)
    tmp36 = tl.where(tmp24, tmp35, 0.0)
    tmp37 = tl.where(tmp24, tmp36, tmp34)
    tmp38 = tl.where(tmp21, tmp22, tmp37)
    tmp39 = tl.where(tmp16, tmp18, tmp38)
    tmp40 = tl.where(tmp5, tmp14, tmp39)
    tl.store(out_ptr0 + (x2 + (513*y3)), tmp40, xmask & ymask)
''')
