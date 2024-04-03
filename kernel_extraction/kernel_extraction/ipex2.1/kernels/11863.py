

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/6o/c6ot6fhsgkyvwfvk7i2x3o5ovbmgal7e5huranttuuskeir6vd6f.py
# Source Nodes: [clone_1, iadd_52, mul_100, mul_145, mul_99, neg_54, setitem_104, sub_46, sub_47, truediv_73, truediv_74], Original ATen: [aten.clone, aten.copy, aten.div, aten.mul, aten.neg, aten.select_scatter, aten.slice, aten.slice_scatter, aten.sub]
# clone_1 => clone_1
# iadd_52 => select_scatter_138, select_scatter_139, slice_761, slice_762
# mul_100 => mul_103
# mul_145 => mul_148
# mul_99 => mul_102
# neg_54 => neg_54
# setitem_104 => copy_104, select_scatter_137, slice_scatter_70, slice_scatter_71, slice_scatter_72
# sub_46 => sub_46
# sub_47 => sub_47
# truediv_73 => div_70
# truediv_74 => div_71
triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_81 = async_compile.triton('triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_81', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_81', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_copy_div_mul_neg_select_scatter_slice_slice_scatter_sub_81(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3246048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3) % 26
    x0 = xindex % 3
    x5 = (xindex // 78)
    x3 = (xindex // 15912)
    x2 = (xindex // 78) % 204
    x6 = (xindex // 3) % 5304
    x7 = xindex
    tmp5 = tl.load(in_ptr0 + (x5), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (x0 + (78*x5)), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr2 + (x7), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tmp3 == tmp1
    tmp6 = x3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.full([1], 202, tl.int64)
    tmp10 = tmp6 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = x2
    tmp13 = tmp12 >= tmp7
    tmp14 = tmp12 < tmp9
    tmp15 = tmp13 & tmp14
    tmp16 = tmp15 & tmp11
    tmp17 = tl.load(in_ptr1 + ((-10452) + (26*x2) + (5200*x3)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr2 + (x0 + (78*x5)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tl.where(tmp16, tmp19, 0.0)
    tmp21 = tl.load(in_ptr2 + (x0 + (78*x5)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.where(tmp15, tmp20, tmp21)
    tmp23 = tl.where(tmp11, tmp22, 0.0)
    tmp25 = tl.where(tmp11, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp5, tmp25)
    tmp27 = tl.load(in_ptr1 + ((-10452) + x6 + (5200*x3)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (x7), tmp16 & xmask, other=0.0)
    tmp29 = tl.where(tmp4, tmp27, tmp28)
    tmp30 = tl.where(tmp16, tmp29, 0.0)
    tmp31 = tl.load(in_ptr2 + (x7), tmp11 & xmask, other=0.0)
    tmp32 = tl.where(tmp15, tmp30, tmp31)
    tmp33 = tl.where(tmp11, tmp32, 0.0)
    tmp35 = tl.where(tmp11, tmp33, tmp34)
    tmp36 = tl.where(tmp2, tmp26, tmp35)
    tl.store(out_ptr0 + (x7), tmp36, xmask)
''')
