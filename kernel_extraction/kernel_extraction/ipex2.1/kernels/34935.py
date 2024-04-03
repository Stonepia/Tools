

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/wy/cwyqjpck4kawpkgh4uozrcrzpljqd7nalxuxxcw24qqnjmnpcj7e.py
# Source Nodes: [abs_6, add_19, add_20, mul_52, mul_56, mul_57, neg_9, setitem_15, tanh_3, truediv_13], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.tanh]
# abs_6 => abs_6
# add_19 => add_22
# add_20 => add_23
# mul_52 => mul_52
# mul_56 => mul_56
# mul_57 => mul_57
# neg_9 => neg_9
# setitem_15 => copy_15, select_scatter_7
# tanh_3 => tanh_3
# truediv_13 => div_13
triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_tanh_5 = async_compile.triton('triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_tanh_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_tanh_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_mul_neg_select_scatter_tanh_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2090400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x4 = (xindex // 2)
    x3 = (xindex // 10400)
    x5 = (xindex // 2) % 5200
    x2 = (xindex // 52) % 200
    x8 = xindex
    tmp3 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (5356 + x5 + (5304*x3)), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr3 + (21426 + x0 + (4*x5) + (21216*x3)), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.abs(tmp3)
    tmp5 = -tmp4
    tmp6 = tl.full([1], 0.001, tl.float64)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 / tmp6
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = tl.full([1], 1.0, tl.float64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0.5, tl.float64)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13 * tmp3
    tmp16 = tmp14 * tmp15
    tmp17 = 1 + x3
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 >= tmp18
    tmp20 = tl.full([1], 202, tl.int64)
    tmp21 = tmp17 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = 2 + x2
    tmp24 = tl.full([1], 2, tl.int64)
    tmp25 = tmp23 >= tmp24
    tmp26 = tmp23 < tmp20
    tmp27 = tmp25 & tmp26
    tmp28 = tmp27 & tmp22
    tmp29 = tl.full([1], 0, tl.int32)
    tmp30 = tmp1 == tmp29
    tmp31 = tl.load(in_ptr2 + (x4), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.abs(tmp31)
    tmp33 = -tmp32
    tmp34 = tmp33 + tmp6
    tmp35 = tmp34 / tmp6
    tmp36 = libdevice.tanh(tmp35)
    tmp37 = tmp36 + tmp10
    tmp38 = tmp37 * tmp12
    tmp39 = tmp38 * tmp31
    tmp40 = tl.load(in_ptr1 + (5356 + x5 + (5304*x3)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr3 + (21424 + x0 + (4*x5) + (21216*x3)), tmp28 & xmask, other=0.0)
    tmp43 = tl.where(tmp2, tmp41, tmp42)
    tmp44 = tl.load(in_ptr3 + (21426 + x0 + (4*x5) + (21216*x3)), tmp28 & xmask, other=0.0)
    tmp45 = tl.where(tmp30, tmp43, tmp44)
    tmp46 = tl.where(tmp28, tmp45, 0.0)
    tmp47 = tl.load(in_ptr3 + (21426 + x0 + (4*x5) + (21216*x3)), tmp22 & xmask, other=0.0)
    tmp48 = tl.where(tmp27, tmp46, tmp47)
    tmp49 = tl.where(tmp22, tmp48, 0.0)
    tmp51 = tl.where(tmp22, tmp49, tmp50)
    tmp52 = tl.where(tmp2, tmp16, tmp51)
    tl.store(out_ptr0 + (x8), tmp52, xmask)
''')
