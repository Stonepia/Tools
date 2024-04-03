

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/5m/c5m4vz3h4pgjx3ql2clhd6okimkh4hofbyi6yv6qpdkybge5r3ah.py
# Source Nodes: [setitem_8], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_8 => copy_8, slice_scatter_29
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

@pointwise(size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256) % 3
    y2 = (yindex // 768)
    y5 = yindex
    tmp15 = tl.load(in_ptr2 + (x3 + (513*y0) + (131328*y2)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x3
    tmp1 = tl.full([1, 1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 256 + ((x3 + (513*y0)) // 512)
    tmp4 = tl.full([1, 1], 512, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (256 + (256*y1) + (256*((131072 + x3 + (513*y0)) // 262656)) + (1024*y2) + (1024*((131072 + x3 + (513*y0) + (262656*y1)) // 787968)) + ((x3 + (513*y0)) // 512)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + ((256*y1) + (256*((131072 + x3 + (513*y0)) // 262656)) + (1024*y2) + (1024*((131072 + x3 + (513*y0) + (262656*y1)) // 787968)) + ((x3 + (513*y0)) % 512)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.where(tmp6, tmp9, 0.0)
    tmp11 = tl.where(tmp2, tmp10, 0.0)
    tmp12 = 1 + y1
    tmp13 = tl.full([1, 1], 3, tl.int32)
    tmp14 = tmp12 == tmp13
    tmp16 = tl.full([1, 1], 3, tl.int64)
    tmp17 = tmp12 < tmp16
    tmp18 = tl.broadcast_to(x3, [XBLOCK, YBLOCK])
    tmp19 = tmp18 >= tmp1
    tmp20 = tmp19 & tmp17
    tmp21 = ((262400 + x3 + (513*y0)) // 512) % 513
    tmp22 = tmp21 < tmp4
    tmp23 = tmp22 & tmp20
    tmp24 = tl.load(in_ptr0 + ((256*(((262400 + x3 + (513*y0) + (262656*y1)) // 262656) % 3)) + (1024*(((262400 + x3 + (513*y0) + (262656*y1) + (787968*y2)) // 787968) % 4)) + (((262400 + x3 + (513*y0)) // 512) % 513)), tmp23 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + ((256*(((262400 + x3 + (513*y0) + (262656*y1)) // 262656) % 3)) + (1024*(((262400 + x3 + (513*y0) + (262656*y1) + (787968*y2)) // 787968) % 4)) + ((262400 + x3 + (513*y0)) % 512)), tmp23 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp24 * tmp25
    tmp27 = tl.where(tmp23, tmp26, 0.0)
    tmp28 = tl.where(tmp20, tmp27, 0.0)
    tmp29 = 0.0
    tmp30 = tl.where(tmp19, tmp28, tmp29)
    tmp31 = tl.where(tmp17, tmp30, 0.0)
    tmp32 = tl.where(tmp17, tmp31, tmp29)
    tmp33 = tl.where(tmp14, tmp15, tmp32)
    tmp34 = tl.where(tmp2, tmp11, tmp33)
    tl.store(out_ptr0 + (x3 + (513*y5)), tmp34, xmask)
''')
