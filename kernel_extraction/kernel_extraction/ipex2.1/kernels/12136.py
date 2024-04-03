

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/wy/cwyuwffek7zjwl6pyxbp3d3skvlfrki3umra3tfp5m6aluy3r6xj.py
# Source Nodes: [iadd_1, iadd_3, iadd_4, mul_15, mul_17, mul_18, neg_3, neg_5, truediv_11, truediv_12], Original ATen: [aten.add, aten.div, aten.mul, aten.neg]
# iadd_1 => add_10
# iadd_3 => add_12
# iadd_4 => add_13
# mul_15 => mul_18
# mul_17 => mul_20
# mul_18 => mul_21
# neg_3 => neg_3
# neg_5 => neg_5
# truediv_11 => div_8
# truediv_12 => div_9
triton_poi_fused_add_div_mul_neg_5 = async_compile.triton('triton_poi_fused_add_div_mul_neg_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp64', 1: '*i64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_neg_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_mul_neg_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp0 = tl.load(in_ptr0 + (1 + (26*x2)), xmask)
    tmp1 = tl.load(in_ptr1 + (410 + x0 + (204*x1)), xmask)
    tmp9 = tl.load(in_ptr2 + (1 + (26*x2)), xmask)
    tmp16 = tl.load(in_ptr0 + (26*x2), xmask)
    tmp19 = tl.load(in_ptr3 + (26*x2), xmask)
    tmp22 = tl.load(in_ptr4 + (2 + (26*x2)), xmask)
    tmp27 = tl.load(in_ptr2 + (2 + (26*x2)), xmask)
    tmp40 = tl.load(in_ptr4 + (1 + (26*x2)), xmask)
    tmp45 = tl.load(in_ptr0 + (2 + (26*x2)), xmask)
    tmp48 = tl.load(in_ptr3 + (1 + (26*x2)), xmask)
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 - tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp2 >= tmp3
    tmp7 = tmp5 & tmp6
    tmp8 = tmp7.to(tl.float64)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 == tmp3
    tmp12 = tmp5 & tmp11
    tmp13 = tmp12 == 0
    tmp14 = tmp13.to(tl.float64)
    tmp15 = tmp10 * tmp14
    tmp17 = tmp15 / tmp16
    tmp18 = -tmp17
    tmp20 = tmp18 * tmp19
    tmp21 = tmp0 + tmp20
    tmp23 = tl.full([1], 2, tl.int64)
    tmp24 = tmp23 >= tmp3
    tmp25 = tmp5 & tmp24
    tmp26 = tmp25.to(tl.float64)
    tmp28 = tmp26 * tmp27
    tmp29 = tmp23 == tmp3
    tmp30 = tmp5 & tmp29
    tmp31 = tmp30 == 0
    tmp32 = tmp31.to(tl.float64)
    tmp33 = tmp28 * tmp32
    tmp34 = tl.full([1], 1, tl.int32)
    tmp35 = tmp34 == tmp34
    tmp36 = tl.where(tmp35, tmp21, tmp0)
    tmp37 = tl.where(tmp35, tmp36, tmp36)
    tmp38 = tmp33 / tmp37
    tmp39 = -tmp38
    tmp41 = tmp39 * tmp40
    tmp42 = tmp22 + tmp41
    tmp43 = tl.full([1], 2, tl.int32)
    tmp44 = tmp43 == tmp34
    tmp46 = tl.where(tmp44, tmp21, tmp45)
    tmp47 = tl.where(tmp44, tmp36, tmp46)
    tmp49 = tmp39 * tmp48
    tmp50 = tmp47 + tmp49
    tl.store(out_ptr0 + (x2), tmp21, xmask)
    tl.store(out_ptr1 + (x2), tmp42, xmask)
    tl.store(out_ptr2 + (x2), tmp50, xmask)
''')
