

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/fj/cfjzcesah7i23mttbvhldxg5ezl7znn22kn4bm2r7znsqibez2fa.py
# Source Nodes: [iadd_15], Original ATen: [aten.slice_scatter]
# iadd_15 => slice_scatter_181
triton_poi_fused_slice_scatter_49 = async_compile.triton('triton_poi_fused_slice_scatter_49', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_49', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1060800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x2 = (xindex // 5304)
    x3 = xindex % 5304
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-52) + x3 + (5200*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = 2 + x2
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp9 & tmp10
    tmp12 = tl.load(in_ptr1 + (x4), tmp11 & xmask, other=0.0)
    tmp13 = tl.where(tmp11, tmp12, 0.0)
    tmp14 = tmp5 & tmp11
    tmp15 = tl.load(in_ptr2 + ((-52) + x3 + (5200*x2)), tmp14 & xmask, other=0.0)
    tmp16 = tl.where(tmp14, tmp15, 0.0)
    tmp17 = tmp11 & tmp11
    tmp18 = tl.load(in_ptr3 + (x4), tmp17 & xmask, other=0.0)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tl.full([1], 0.0, tl.float64)
    tmp21 = tl.where(tmp11, tmp19, tmp20)
    tmp22 = tl.where(tmp5, tmp16, tmp21)
    tmp23 = tl.where(tmp11, tmp22, 0.0)
    tmp24 = tl.load(in_ptr3 + (x4), tmp11 & xmask, other=0.0)
    tmp25 = tl.where(tmp11, tmp24, 0.0)
    tmp26 = tl.where(tmp11, tmp25, tmp20)
    tmp27 = tl.where(tmp11, tmp23, tmp26)
    tmp28 = tl.where(tmp11, tmp13, tmp27)
    tmp29 = tl.where(tmp5, tmp7, tmp28)
    tl.store(out_ptr0 + (x4), tmp29, xmask)
''')
