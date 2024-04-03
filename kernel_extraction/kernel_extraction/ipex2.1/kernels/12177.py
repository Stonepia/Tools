

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/sn/csnv3idjf3ree4lpccrsmy2vbzemv52adn4xorj4nkgrp2bgn6m2.py
# Source Nodes: [iadd_41, iadd_43, mul_55, neg_43, setitem_48, setitem_50, truediv_31], Original ATen: [aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter]
# iadd_41 => add_50, select_scatter_84
# iadd_43 => select_scatter_88
# mul_55 => mul_58
# neg_43 => neg_43
# setitem_48 => copy_48, select_scatter_85
# setitem_50 => copy_50, select_scatter_89
# truediv_31 => div_28
triton_poi_fused_add_copy_div_mul_neg_select_scatter_46 = async_compile.triton('triton_poi_fused_add_copy_div_mul_neg_select_scatter_46', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_neg_select_scatter_46', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_div_mul_neg_select_scatter_46(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp4 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (21 + (26*x1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (22 + (26*x1)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 22, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp1 == tmp1
    tmp5 = tl.full([1], 21, tl.int32)
    tmp6 = tmp1 == tmp5
    tmp7 = tmp5 == tmp5
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp12 = tl.where(tmp6, tmp8, tmp11)
    tmp13 = tl.where(tmp6, tmp10, tmp12)
    tmp14 = tl.where(tmp3, tmp4, tmp13)
    tmp15 = tmp0 == tmp5
    tmp17 = tl.where(tmp15, tmp8, tmp16)
    tmp18 = tl.where(tmp15, tmp10, tmp17)
    tmp19 = tl.where(tmp2, tmp4, tmp18)
    tmp20 = tl.where(tmp2, tmp14, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
''')