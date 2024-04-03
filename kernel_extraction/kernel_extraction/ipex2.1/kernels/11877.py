

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/24/c24ufhbz2gdkabpk2a4cg2ymeyhaxiq3fabfq7wzzym7h3vprgvs.py
# Source Nodes: [iadd_1, iadd_3, mul_15, mul_17, mul_18, neg_3, neg_5, truediv_11, truediv_12], Original ATen: [aten.add, aten.div, aten.mul, aten.neg]
# iadd_1 => add_10
# iadd_3 => add_12
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

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_neg_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
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
    tmp9 = tl.load(in_ptr2 + (1 + (26*x2)), xmask).to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (26*x2), xmask)
    tmp20 = tl.load(in_ptr3 + (26*x2), xmask).to(tl.float32)
    tmp28 = tl.load(in_ptr2 + (2 + (26*x2)), xmask).to(tl.float32)
    tmp42 = tl.load(in_ptr4 + (1 + (26*x2)), xmask).to(tl.float32)
    tmp48 = tl.load(in_ptr0 + (2 + (26*x2)), xmask)
    tmp51 = tl.load(in_ptr3 + (1 + (26*x2)), xmask).to(tl.float32)
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 - tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp2 >= tmp3
    tmp7 = tmp5 & tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 == tmp3
    tmp12 = tmp5 & tmp11
    tmp13 = tmp12 == 0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp10 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 / tmp17
    tmp19 = -tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 * tmp21
    tmp23 = tmp0 + tmp22
    tmp24 = tl.full([1], 2, tl.int64)
    tmp25 = tmp24 >= tmp3
    tmp26 = tmp5 & tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp24 == tmp3
    tmp31 = tmp5 & tmp30
    tmp32 = tmp31 == 0
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp29 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 == tmp36
    tmp38 = tl.where(tmp37, tmp23, tmp0)
    tmp39 = tl.where(tmp37, tmp38, tmp38)
    tmp40 = tmp35 / tmp39
    tmp41 = -tmp40
    tmp43 = tl.where(tmp37, tmp42, tmp42)
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp41 * tmp44
    tmp46 = tl.full([1], 2, tl.int32)
    tmp47 = tmp46 == tmp36
    tmp49 = tl.where(tmp47, tmp23, tmp48)
    tmp50 = tl.where(tmp47, tmp38, tmp49)
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp41 * tmp52
    tmp54 = tmp50 + tmp53
    tl.store(out_ptr0 + (x2), tmp23, xmask)
    tl.store(out_ptr1 + (x2), tmp45, xmask)
    tl.store(out_ptr2 + (x2), tmp54, xmask)
''')
