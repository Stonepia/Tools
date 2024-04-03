

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/t4/ct4ndelz3phlrgfb6dhobneoeloabiv7i6jjikdaepmhkczgqj7x.py
# Source Nodes: [iadd_14, mul_28, neg_15, truediv_17], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.neg]
# iadd_14 => add_23, convert_element_type_6
# mul_28 => mul_31
# neg_15 => neg_15
# truediv_17 => div_14
triton_poi_fused__to_copy_add_div_mul_neg_15 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_neg_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_neg_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_neg_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200
    x1 = (xindex // 200)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (410 + x0 + (204*x1)), xmask)
    tmp9 = tl.load(in_ptr1 + (7 + (26*x2)), xmask).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (6 + (26*x2)), xmask)
    tmp24 = tl.load(in_ptr3 + (5 + (26*x2)), xmask).to(tl.float32)
    tmp25 = tl.load(in_ptr3 + (6 + (26*x2)), xmask).to(tl.float32)
    tmp28 = tl.load(in_ptr4 + (x2), xmask)
    tmp38 = tl.load(in_ptr3 + (7 + (26*x2)), xmask).to(tl.float32)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 >= tmp3
    tmp5 = tl.full([1], 7, tl.int64)
    tmp6 = tmp5 >= tmp2
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 == tmp2
    tmp12 = tmp4 & tmp11
    tmp13 = tmp12 == 0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp10 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 / tmp17
    tmp19 = -tmp18
    tmp20 = tl.full([1], 6, tl.int32)
    tmp21 = tmp20 == tmp20
    tmp22 = tl.full([1], 5, tl.int32)
    tmp23 = tmp20 == tmp22
    tmp26 = tl.where(tmp23, tmp24, tmp25)
    tmp27 = tmp26.to(tl.float32)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.where(tmp21, tmp30, tmp26)
    tmp32 = tl.where(tmp21, tmp31, tmp31)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp19 * tmp33
    tmp35 = tl.full([1], 7, tl.int32)
    tmp36 = tmp35 == tmp20
    tmp37 = tmp35 == tmp22
    tmp39 = tl.where(tmp37, tmp24, tmp38)
    tmp40 = tl.where(tmp36, tmp30, tmp39)
    tmp41 = tl.where(tmp36, tmp31, tmp40)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 + tmp34
    tmp44 = tmp43.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp44, xmask)
''')