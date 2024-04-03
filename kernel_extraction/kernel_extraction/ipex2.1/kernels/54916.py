

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/xs/cxsf56e6z43tlqepcs5ld7ce5o2gwmykz4xc76jfhars474sf4vs.py
# Source Nodes: [setitem_7], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_7 => copy_7, slice_scatter_26
triton_poi_fused_copy_slice_scatter_15 = async_compile.triton('triton_poi_fused_copy_slice_scatter_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_15(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 256, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ((656384 + x2 + (513*y0)) // 512) % 513
    tmp4 = tl.full([1, 1], 512, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((256*((656384 + x2 + (513*y0)) // 262656)) + (1024*y1) + (1024*((656384 + x2 + (513*y0)) // 787968)) + (((656384 + x2 + (513*y0)) // 512) % 513)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + ((256*((656384 + x2 + (513*y0)) // 262656)) + (1024*y1) + (1024*((656384 + x2 + (513*y0)) // 787968)) + ((x2 + (513*y0)) % 512)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.where(tmp6, tmp9, 0.0)
    tmp11 = tl.where(tmp2, tmp10, 0.0)
    tmp12 = tl.full([1, 1], 3, tl.int64)
    tmp13 = tmp12 < tmp12
    tmp14 = tl.broadcast_to(x2, [XBLOCK, YBLOCK])
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp15 & tmp13
    tmp17 = ((787712 + x2 + (513*y0)) // 512) % 513
    tmp18 = tmp17 < tmp4
    tmp19 = tmp18 & tmp16
    tmp20 = tl.load(in_ptr0 + ((256*(((787712 + x2 + (513*y0)) // 262656) % 3)) + (1024*(((787712 + x2 + (513*y0) + (787968*y1)) // 787968) % 4)) + (((787712 + x2 + (513*y0)) // 512) % 513)), tmp19 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr1 + ((256*(((787712 + x2 + (513*y0)) // 262656) % 3)) + (1024*(((787712 + x2 + (513*y0) + (787968*y1)) // 787968) % 4)) + ((787712 + x2 + (513*y0)) % 512)), tmp19 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.where(tmp19, tmp22, 0.0)
    tmp24 = tl.where(tmp16, tmp23, 0.0)
    tmp25 = 0.0
    tmp26 = tl.where(tmp15, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp26, 0.0)
    tmp28 = tl.where(tmp13, tmp27, tmp25)
    tmp29 = tl.where(tmp2, tmp11, tmp28)
    tl.store(out_ptr0 + (x2 + (513*y3)), tmp29, xmask)
''')
