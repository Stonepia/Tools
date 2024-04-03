

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/z5/cz533co5xruejmm5ogswdyib3jbynjqokbgdb6otxvmg4ms4vofs.py
# Source Nodes: [add_12, mul_117, mul_124, mul_125, mul_130, mul_95, mul_96, mul_97, setitem_89, setitem_90, setitem_96, setitem_97, sub_39, truediv_64, zeros_like], Original ATen: [aten.add, aten.copy, aten.div, aten.mul, aten.select_scatter, aten.slice, aten.slice_scatter, aten.sub, aten.zeros_like]
# add_12 => add_64
# mul_117 => mul_120
# mul_124 => mul_127
# mul_125 => mul_128
# mul_130 => mul_133
# mul_95 => mul_98
# mul_96 => mul_99
# mul_97 => mul_100
# setitem_89 => copy_89, slice_241, slice_scatter_32, slice_scatter_33, slice_scatter_34
# setitem_90 => copy_90, select_scatter_134, slice_251, slice_scatter_35, slice_scatter_36
# setitem_96 => copy_96
# setitem_97 => copy_97, slice_scatter_52, slice_scatter_53, slice_scatter_54
# sub_39 => sub_39
# truediv_64 => div_61
# zeros_like => full
triton_poi_fused_add_copy_div_mul_select_scatter_slice_slice_scatter_sub_zeros_like_71 = async_compile.triton('triton_poi_fused_add_copy_div_mul_select_scatter_slice_slice_scatter_sub_zeros_like_71', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_select_scatter_slice_slice_scatter_sub_zeros_like_71', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_div_mul_select_scatter_slice_slice_scatter_sub_zeros_like_71(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1082016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5304)
    x1 = (xindex // 26) % 204
    x3 = xindex
    x4 = xindex % 5304
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr0 + (x1), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (3*x3), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.load(in_ptr2 + (78 + (3*x3)), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (3*x3), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 * tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.load(in_ptr3 + ((-10478) + x4 + (5226*x2)), tmp11 & xmask, other=0.0)
    tmp23 = tmp22 * tmp19
    tmp24 = tmp21 - tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tl.where(tmp11, tmp25, 0.0)
    tmp27 = 0.0
    tmp28 = tl.where(tmp10, tmp26, tmp27)
    tmp29 = tl.where(tmp5, tmp28, 0.0)
    tmp30 = tl.where(tmp5, tmp29, tmp27)
    tl.store(out_ptr0 + (x3), tmp30, xmask)
''')
