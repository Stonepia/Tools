

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/se/csepcopqegmksczmgtmwqawb6d6f2d6wjqjjkqvswhqkx52ffcud.py
# Source Nodes: [and__1, ge_1, mul_86, mul_87, mul_88, setitem_80, setitem_81, setitem_82, setitem_83, setitem_84, sub_22, sub_23, sub_24, truediv_58, truediv_59, truediv_60, where_2], Original ATen: [aten.bitwise_and, aten.copy, aten.div, aten.ge, aten.mul, aten.select_scatter, aten.sub, aten.where]
# and__1 => bitwise_and_1
# ge_1 => ge_1
# mul_86 => mul_89
# mul_87 => mul_90
# mul_88 => mul_91
# setitem_80 => copy_80, select_scatter_126
# setitem_81 => copy_81, select_scatter_127
# setitem_82 => copy_82, select_scatter_128
# setitem_83 => copy_83, select_scatter_129
# setitem_84 => copy_84
# sub_22 => sub_22
# sub_23 => sub_23
# sub_24 => sub_24
# truediv_58 => div_55
# truediv_59 => div_56
# truediv_60 => div_57
# where_2 => where_2
triton_poi_fused_bitwise_and_copy_div_ge_mul_select_scatter_sub_where_65 = async_compile.triton('triton_poi_fused_bitwise_and_copy_div_ge_mul_select_scatter_sub_where_65', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_copy_div_ge_mul_select_scatter_sub_where_65', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bitwise_and_copy_div_ge_mul_select_scatter_sub_where_65(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 200
    x2 = (xindex // 5200)
    x0 = xindex % 26
    x3 = (xindex // 26)
    x4 = xindex
    x5 = xindex % 5200
    tmp0 = tl.load(in_ptr0 + (410 + x1 + (204*x2)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp25 = tl.load(in_ptr5 + (31981 + (3*x5) + (15912*x2)), xmask)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 >= tmp3
    tmp5 = x0
    tmp6 = tmp5 >= tmp2
    tmp7 = tmp4 & tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = tmp5 == tmp8
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp5 == tmp11
    tmp14 = tl.full([1], 2, tl.int32)
    tmp15 = tmp5 == tmp14
    tmp17 = tl.full([1], 3, tl.int32)
    tmp18 = tmp5 == tmp17
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp15, tmp16, tmp21)
    tmp23 = tl.where(tmp12, tmp13, tmp22)
    tmp24 = tl.where(tmp9, tmp10, tmp23)
    tmp26 = tl.where(tmp7, tmp24, tmp25)
    tl.store(in_out_ptr0 + (x4), tmp26, xmask)
''')
