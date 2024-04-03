

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/7r/c7r6gz52kelh7ebwoomzrkgftqruotpjug5kdwp53ndpr5aaoskk.py
# Source Nodes: [iadd_10, iadd_8, mul_22, neg_9, setitem_15, setitem_17, truediv_14], Original ATen: [aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter]
# iadd_10 => select_scatter_22
# iadd_8 => add_17, select_scatter_18
# mul_22 => mul_25
# neg_9 => neg_9
# setitem_15 => copy_15, select_scatter_19
# setitem_17 => copy_17, select_scatter_23
# truediv_14 => div_11
triton_poi_fused_add_copy_div_mul_neg_select_scatter_12 = async_compile.triton('triton_poi_fused_add_copy_div_mul_neg_select_scatter_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_neg_select_scatter_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_div_mul_neg_select_scatter_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp4 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (4 + (26*x1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (5 + (26*x1)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 5, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp1 == tmp1
    tmp5 = tl.full([1], 4, tl.int32)
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
