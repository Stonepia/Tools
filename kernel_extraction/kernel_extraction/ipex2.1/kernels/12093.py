

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/fh/cfhmleb3rf3gwlv77pviujnaofrdzgmwbyyyqaugl4vzwj6awvct.py
# Source Nodes: [iadd_49, mul_63, mul_65, neg_51, setitem_59, sub_1, truediv_35, truediv_37], Original ATen: [aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.sub]
# iadd_49 => add_58
# mul_63 => mul_66
# mul_65 => mul_68
# neg_51 => neg_51
# setitem_59 => copy_59
# sub_1 => sub_1
# truediv_35 => div_32
# truediv_37 => div_34
triton_poi_fused_add_copy_div_mul_neg_sub_53 = async_compile.triton('triton_poi_fused_add_copy_div_mul_neg_sub_53', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp64', 1: '*i64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_neg_sub_53', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_div_mul_neg_sub_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp0 = tl.load(in_ptr0 + (25 + (26*x2)), xmask)
    tmp1 = tl.load(in_ptr1 + (410 + x0 + (204*x1)), xmask)
    tmp10 = tl.load(in_ptr2 + (25 + (26*x2)), xmask)
    tmp17 = tl.load(in_ptr0 + (24 + (26*x2)), xmask)
    tmp20 = tl.load(in_ptr3 + (24 + (26*x2)), xmask)
    tmp23 = tl.load(in_ptr4 + (24 + (26*x2)), xmask)
    tmp26 = tl.load(in_ptr4 + (25 + (26*x2)), xmask)
    tmp30 = tl.load(in_ptr5 + (25 + (26*x2)), xmask)
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 - tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 25, tl.int64)
    tmp7 = tmp6 >= tmp3
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8.to(tl.float64)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 == tmp3
    tmp13 = tmp5 & tmp12
    tmp14 = tmp13 == 0
    tmp15 = tmp14.to(tl.float64)
    tmp16 = tmp11 * tmp15
    tmp18 = tmp16 / tmp17
    tmp19 = -tmp18
    tmp21 = tmp19 * tmp20
    tmp22 = tmp0 + tmp21
    tmp24 = tl.full([1], 25, tl.int32)
    tmp25 = tmp24 == tmp24
    tmp27 = tl.where(tmp25, tmp22, tmp0)
    tmp28 = tl.where(tmp25, tmp27, tmp27)
    tmp29 = tmp26 / tmp28
    tmp31 = tl.where(tmp25, tmp29, tmp30)
    tmp32 = tmp20 * tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tl.full([1], 24, tl.int32)
    tmp35 = tmp34 == tmp24
    tmp36 = tl.where(tmp35, tmp22, tmp17)
    tmp37 = tl.where(tmp35, tmp27, tmp36)
    tmp38 = tmp33 / tmp37
    tl.store(out_ptr0 + (x2), tmp22, xmask)
    tl.store(out_ptr1 + (x2), tmp38, xmask)
''')
