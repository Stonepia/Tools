

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ua/cuaegf37wlgthxu6i5sahx3cs75xz5dl673lnlbeypoopoqdqy4i.py
# Source Nodes: [grid_sample, grid_sample_1], Original ATen: [aten.grid_sampler_2d]
# grid_sample => full_default, full_default_2
# grid_sample_1 => add_49, add_50, add_51, add_52, convert_element_type_149, convert_element_type_152, floor_2, floor_3, ge_10, ge_13, ge_8, ge_9, logical_and_12, logical_and_13, logical_and_14, logical_and_16, logical_and_17, logical_and_18, logical_and_19, logical_and_20, logical_and_22, logical_and_23, lt_10, lt_11, lt_14, lt_9, mul_100, mul_101, mul_96, mul_97, mul_98, mul_99, sub_47, sub_48, sub_49, sub_52, where_38, where_41, where_44, where_45, where_46, where_47
triton_poi_fused_grid_sampler_2d_32 = async_compile.triton('triton_poi_fused_grid_sampler_2d_32', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*i64', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_grid_sampler_2d_32(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 743424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), None)
    tmp10 = tl.load(in_ptr0 + (1 + (2*x0)), None)
    tmp1 = 176.0
    tmp2 = tmp0 * tmp1
    tmp3 = 175.5
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.floor(tmp4)
    tmp6 = 0.0
    tmp7 = tmp5 >= tmp6
    tmp8 = 352.0
    tmp9 = tmp5 < tmp8
    tmp11 = tmp10 * tmp1
    tmp12 = tmp11 + tmp3
    tmp13 = libdevice.floor(tmp12)
    tmp14 = tmp13 >= tmp6
    tmp15 = tmp13 < tmp8
    tmp16 = tmp14 & tmp15
    tmp17 = tmp9 & tmp16
    tmp18 = tmp7 & tmp17
    tmp19 = 1.0
    tmp20 = tmp5 + tmp19
    tmp21 = tmp20 - tmp4
    tmp22 = tmp13 + tmp19
    tmp23 = tmp22 - tmp12
    tmp24 = tmp21 * tmp23
    tmp25 = tl.where(tmp18, tmp24, tmp6)
    tmp26 = tmp20 >= tmp6
    tmp27 = tmp20 < tmp8
    tmp28 = tmp27 & tmp16
    tmp29 = tmp26 & tmp28
    tmp30 = tmp4 - tmp5
    tmp31 = tmp30 * tmp23
    tmp32 = tl.where(tmp29, tmp31, tmp6)
    tmp33 = tmp22 >= tmp6
    tmp34 = tmp22 < tmp8
    tmp35 = tmp33 & tmp34
    tmp36 = tmp9 & tmp35
    tmp37 = tmp7 & tmp36
    tmp38 = tmp12 - tmp13
    tmp39 = tmp21 * tmp38
    tmp40 = tl.where(tmp37, tmp39, tmp6)
    tmp41 = tmp27 & tmp35
    tmp42 = tmp26 & tmp41
    tmp43 = tmp22.to(tl.int64)
    tmp44 = tl.full([1], 0, tl.int64)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = tmp20.to(tl.int64)
    tmp47 = tl.where(tmp42, tmp46, tmp44)
    tmp48 = tmp30 * tmp38
    tmp49 = tl.where(tmp42, tmp48, tmp6)
    tl.store(out_ptr0 + (x0), tmp25, None)
    tl.store(out_ptr1 + (x0), tmp32, None)
    tl.store(out_ptr2 + (x0), tmp40, None)
    tl.store(out_ptr3 + (x0), tmp45, None)
    tl.store(out_ptr4 + (x0), tmp47, None)
    tl.store(out_ptr5 + (x0), tmp49, None)
''')
