

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/zt/cztz2j3v772tatv747ylcqvqvz4ohxg5dsly5lahz4lyfw4dfcra.py
# Source Nodes: [iadd_54, mul_3, neg_57, sub_49, truediv_77], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.sub]
# iadd_54 => add_70, select_scatter_144
# mul_3 => mul_4
# neg_57 => neg_57
# sub_49 => sub_49
# truediv_77 => div_74
triton_poi_fused_add_div_mul_neg_select_scatter_sub_86 = async_compile.triton('triton_poi_fused_add_div_mul_neg_select_scatter_sub_86', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_neg_select_scatter_sub_86', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_mul_neg_select_scatter_sub_86(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 124848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x3 = (xindex // 3)
    x2 = (xindex // 612)
    x4 = xindex
    tmp12 = tl.load(in_ptr0 + (75 + (78*x3)), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr2 + (25))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp38 = tl.load(in_ptr0 + (75 + x0 + (78*x3)), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 25, tl.int64)
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp3 < tmp3
    tmp7 = tmp5 & tmp6
    tmp8 = tmp1 == tmp1
    tmp9 = tl.load(in_ptr0 + (75 + (78*x3)), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp8, tmp9, tmp9)
    tmp11 = tl.where(tmp7, tmp10, 0.0)
    tmp13 = tl.where(tmp7, tmp11, tmp12)
    tmp14 = x2
    tmp15 = tl.full([1], 2, tl.int64)
    tmp16 = tmp14 >= tmp15
    tmp17 = tl.full([1], 202, tl.int64)
    tmp18 = tmp14 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + ((-10583) + (26*x3)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.where(tmp19, tmp20, 0.0)
    tmp22 = tl.full([1], 0.0, tl.float64)
    tmp23 = tl.where(tmp19, tmp21, tmp22)
    tmp24 = tl.load(in_ptr1 + ((-10584) + (26*x3)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.where(tmp19, tmp24, 0.0)
    tmp26 = tl.where(tmp19, tmp25, tmp22)
    tmp27 = tmp23 - tmp26
    tmp28 = -tmp27
    tmp31 = tl.full([1], 0.5, tl.float64)
    tmp32 = tmp30 * tmp31
    tmp33 = tmp28 / tmp32
    tmp34 = tmp13 + tmp33
    tmp35 = tl.load(in_ptr0 + (75 + x0 + (78*x3)), tmp7 & xmask, other=0.0)
    tmp36 = tl.where(tmp2, tmp9, tmp35)
    tmp37 = tl.where(tmp7, tmp36, 0.0)
    tmp39 = tl.where(tmp7, tmp37, tmp38)
    tmp40 = tl.where(tmp2, tmp34, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''')
