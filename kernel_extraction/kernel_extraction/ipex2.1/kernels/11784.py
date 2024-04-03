

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/a4/ca4j3z74n2cfzfssgtx6k6xpkfezdhd552d7ga5hitvi6fvjvz6o.py
# Source Nodes: [iadd_2, mul_16, neg_3, truediv_11], Original ATen: [aten.add, aten.div, aten.mul, aten.neg]
# iadd_2 => add_11
# mul_16 => mul_19
# neg_3 => neg_3
# truediv_11 => div_8
triton_poi_fused_add_div_mul_neg_2 = async_compile.triton('triton_poi_fused_add_div_mul_neg_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_neg_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_mul_neg_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200
    x1 = (xindex // 200)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (410 + x0 + (204*x1)), xmask)
    tmp11 = tl.load(in_ptr1 + (25 + (26*x2)), xmask)
    tmp12 = tl.load(in_ptr1 + (1 + (26*x2)), xmask)
    tmp15 = tl.load(in_ptr2 + (1 + (26*x2)), xmask)
    tmp22 = tl.load(in_ptr3 + (26*x2), xmask)
    tmp30 = tl.load(in_ptr1 + (26*x2), xmask)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 >= tmp3
    tmp5 = tmp1 >= tmp2
    tmp6 = tmp4 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tl.full([1], 25, tl.int32)
    tmp10 = tmp8 == tmp9
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tmp7 * tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp1 == tmp2
    tmp18 = tmp4 & tmp17
    tmp19 = tmp18 == 0
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 / tmp22
    tmp24 = -tmp23
    tmp25 = tmp3 >= tmp2
    tmp26 = tmp4 & tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tl.full([1], 0, tl.int32)
    tmp29 = tmp28 == tmp9
    tmp31 = tl.where(tmp29, tmp11, tmp30)
    tmp32 = tmp27 * tmp31
    tmp33 = tmp24 * tmp32
    tmp34 = tmp14 + tmp33
    tl.store(out_ptr0 + (x2), tmp34, xmask)
''')
