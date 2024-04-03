

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/7a/c7aks5j3b2ijhffazsxzcbozb3uc56ybbftowlcd7trsdrihu67o.py
# Source Nodes: [iadd_21, iadd_23, mul_35, neg_23, setitem_28, truediv_21], Original ATen: [aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter]
# iadd_21 => add_30, select_scatter_44
# iadd_23 => add_32, select_scatter_48
# mul_35 => mul_38
# neg_23 => neg_23
# setitem_28 => copy_28, select_scatter_45
# truediv_21 => div_18
triton_poi_fused_add_copy_div_mul_neg_select_scatter_26 = async_compile.triton('triton_poi_fused_add_copy_div_mul_neg_select_scatter_26', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_neg_select_scatter_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_div_mul_neg_select_scatter_26(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp6 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (11 + (26*x1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (12 + (26*x1)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 12, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 11, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp5 = tmp3 == tmp3
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp10 = tl.where(tmp4, tmp6, tmp9)
    tmp11 = tl.where(tmp4, tmp8, tmp10)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp6, tmp15)
    tmp17 = tl.where(tmp14, tmp8, tmp16)
    tmp18 = tl.where(tmp2, tmp13, tmp17)
    tl.store(out_ptr0 + (x2), tmp18, xmask)
''')
