

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/7u/c7u4jnbnapd6uhal52wqk7ptxh7viak3mibb7yc7ghmilcozlqep.py
# Source Nodes: [iadd_29, iadd_31, mul_43, mul_45, neg_31, neg_33, setitem_34, setitem_36, truediv_25], Original ATen: [aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter]
# iadd_29 => add_38, select_scatter_60
# iadd_31 => add_40, select_scatter_64
# mul_43 => mul_46
# mul_45 => mul_48
# neg_31 => neg_31
# neg_33 => neg_33
# setitem_34 => copy_34, select_scatter_57
# setitem_36 => copy_36, select_scatter_61
# truediv_25 => div_22
triton_poi_fused_add_copy_div_mul_neg_select_scatter_33 = async_compile.triton('triton_poi_fused_add_copy_div_mul_neg_select_scatter_33', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_neg_select_scatter_33', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_div_mul_neg_select_scatter_33(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (14 + (26*x1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (15 + (26*x1)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 16, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 15, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp6 = tmp4 == tmp4
    tmp8 = tl.full([1], 14, tl.int32)
    tmp9 = tmp4 == tmp8
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.where(tmp6, tmp7, tmp12)
    tmp14 = tmp0 == tmp8
    tmp16 = tl.where(tmp14, tmp10, tmp15)
    tmp17 = tl.where(tmp5, tmp7, tmp16)
    tmp18 = tl.where(tmp5, tmp13, tmp17)
    tmp19 = tl.where(tmp2, tmp3, tmp18)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''')
