

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/3t/c3tmdwxc3ys7uw4kewxzav4wcaswfpaoudk42xyxpkjol66u7m3h.py
# Source Nodes: [lt, mul_90, mul_91, neg_53, truediv_62, where_3], Original ATen: [aten.div, aten.lt, aten.mul, aten.neg, aten.scalar_tensor, aten.where]
# lt => lt
# mul_90 => mul_93
# mul_91 => mul_94
# neg_53 => neg_53
# truediv_62 => div_59
# where_3 => full_default_3, where_3
triton_poi_fused_div_lt_mul_neg_scalar_tensor_where_88 = async_compile.triton('triton_poi_fused_div_lt_mul_neg_scalar_tensor_where_88', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_lt_mul_neg_scalar_tensor_where_88', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_div_lt_mul_neg_scalar_tensor_where_88(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 200)
    x0 = xindex % 200
    x2 = xindex
    tmp20 = tl.load(in_ptr1 + (32056 + (78*x0) + (15912*x1)), xmask)
    tmp27 = tl.load(in_ptr2 + (25))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp0 = 2 + x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2 + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp9 & tmp5
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 == tmp11
    tmp13 = tl.load(in_ptr0 + (25 + (26*x2)), tmp10 & xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (32056 + (78*x0) + (15912*x1)), tmp10 & xmask, other=0.0)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, 0.0)
    tmp17 = tl.load(in_ptr1 + (32056 + (78*x0) + (15912*x1)), tmp5 & xmask, other=0.0)
    tmp18 = tl.where(tmp9, tmp16, tmp17)
    tmp19 = tl.where(tmp5, tmp18, 0.0)
    tmp21 = tl.where(tmp5, tmp19, tmp20)
    tmp22 = 0.0
    tmp23 = tmp21 < tmp22
    tmp24 = -tmp21
    tmp25 = 0.5
    tmp26 = tmp24 * tmp25
    tmp29 = tmp26 * tmp28
    tmp30 = 1.0
    tmp31 = tmp29 / tmp30
    tmp32 = tl.where(tmp23, tmp31, tmp22)
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''')
