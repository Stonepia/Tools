

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/g2/cg2hqkwgq7jf3a5horpqwx6c7y3hgettghy6lole7sfqqv2gsnj2.py
# Source Nodes: [iadd_34, mul_48, neg_35, truediv_27], Original ATen: [aten.add, aten.div, aten.mul, aten.neg]
# iadd_34 => add_43
# mul_48 => mul_51
# neg_35 => neg_35
# truediv_27 => div_24
triton_poi_fused_add_div_mul_neg_35 = async_compile.triton('triton_poi_fused_add_div_mul_neg_35', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*i64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_neg_35', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_mul_neg_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp4 = tl.load(in_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (16 + (26*x2)), xmask)
    tmp7 = tl.load(in_ptr1 + (17 + (26*x2)), xmask)
    tmp10 = tl.load(in_ptr2 + (410 + x0 + (204*x1)), xmask)
    tmp19 = tl.load(in_ptr3 + (17 + (26*x2)), xmask)
    tmp26 = tl.load(in_ptr4 + (16 + (26*x2)), xmask)
    tmp0 = tl.full([1], 17, tl.int32)
    tmp1 = tl.full([1], 16, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp1 == tmp1
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp8 = tl.where(tmp2, tmp4, tmp7)
    tmp9 = tl.where(tmp2, tmp6, tmp8)
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 - tmp11
    tmp13 = tl.full([1], 0, tl.int64)
    tmp14 = tmp12 >= tmp13
    tmp15 = tl.full([1], 17, tl.int64)
    tmp16 = tmp15 >= tmp12
    tmp17 = tmp14 & tmp16
    tmp18 = tmp17.to(tl.float64)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp15 == tmp12
    tmp22 = tmp14 & tmp21
    tmp23 = tmp22 == 0
    tmp24 = tmp23.to(tl.float64)
    tmp25 = tmp20 * tmp24
    tmp27 = tmp25 / tmp26
    tmp28 = -tmp27
    tmp29 = tl.where(tmp3, tmp6, tmp6)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp9 + tmp30
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''')
