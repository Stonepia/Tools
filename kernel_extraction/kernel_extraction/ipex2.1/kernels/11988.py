

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/3j/c3jq5wybhg7a4zqby57cqyd7hs353utiqya76cftqwkjcqfz547v.py
# Source Nodes: [iadd_30, iadd_32, mul_44, mul_46, neg_31, neg_33, setitem_37, truediv_25], Original ATen: [aten._to_copy, aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter]
# iadd_30 => add_39, convert_element_type_14, select_scatter_62
# iadd_32 => add_41, convert_element_type_15, select_scatter_66
# mul_44 => mul_47
# mul_46 => mul_49
# neg_31 => neg_31
# neg_33 => neg_33
# setitem_37 => copy_37, select_scatter_63
# truediv_25 => div_22
triton_poi_fused__to_copy_add_copy_div_mul_neg_select_scatter_32 = async_compile.triton('triton_poi_fused__to_copy_add_copy_div_mul_neg_select_scatter_32', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_copy_div_mul_neg_select_scatter_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_copy_div_mul_neg_select_scatter_32(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp6 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (15 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (16 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 16, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 15, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp5 = tmp3 == tmp3
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp10 = tl.where(tmp4, tmp6, tmp9)
    tmp11 = tl.where(tmp4, tmp8, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tmp14 = -tmp13
    tmp15 = tl.where(tmp5, tmp8, tmp8)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp18 = tmp12 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp0 == tmp3
    tmp22 = tl.where(tmp20, tmp6, tmp21)
    tmp23 = tl.where(tmp20, tmp8, tmp22)
    tmp24 = tl.where(tmp2, tmp19, tmp23)
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''')
