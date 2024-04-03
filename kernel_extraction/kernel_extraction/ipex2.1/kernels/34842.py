

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/rg/crgpqbdn2trokkb72wfrr63ser5xwxzun3wugfinpdbi6milajm6.py
# Source Nodes: [iadd_3], Original ATen: [aten.slice_scatter]
# iadd_3 => slice_scatter_62, slice_scatter_63, slice_scatter_64, slice_scatter_65
triton_poi_fused_slice_scatter_19 = async_compile.triton('triton_poi_fused_slice_scatter_19', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1066104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x2 = (xindex // 5304)
    x3 = xindex % 5304
    x4 = xindex
    x0 = xindex % 26
    tmp29 = tl.load(in_ptr3 + (5304 + x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-52) + x3 + (5200*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = 1 + x2
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tmp8 < tmp3
    tmp12 = tmp10 & tmp11
    tmp13 = tl.load(in_ptr1 + (x4), tmp12 & xmask, other=0.0)
    tmp14 = tl.where(tmp12, tmp13, 0.0)
    tmp15 = tl.load(in_ptr2 + (x4), tmp12 & xmask, other=0.0)
    tmp16 = tl.where(tmp12, tmp15, 0.0)
    tmp17 = tmp5 & tmp12
    tmp18 = x0
    tmp19 = tmp18 >= tmp9
    tmp20 = tmp19 & tmp17
    tmp21 = tl.load(in_ptr3 + (5304 + x4), tmp20 & xmask, other=0.0)
    tmp22 = tl.where(tmp20, tmp21, 0.0)
    tmp23 = tl.load(in_ptr3 + (5304 + x4), tmp17 & xmask, other=0.0)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp17, tmp24, 0.0)
    tmp26 = tl.load(in_ptr3 + (5304 + x4), tmp12 & xmask, other=0.0)
    tmp27 = tl.where(tmp5, tmp25, tmp26)
    tmp28 = tl.where(tmp12, tmp27, 0.0)
    tmp30 = tl.where(tmp12, tmp28, tmp29)
    tmp31 = tl.where(tmp12, tmp16, tmp30)
    tmp32 = tl.where(tmp12, tmp14, tmp31)
    tmp33 = tl.where(tmp5, tmp7, tmp32)
    tl.store(out_ptr0 + (x4), tmp33, xmask)
''')
