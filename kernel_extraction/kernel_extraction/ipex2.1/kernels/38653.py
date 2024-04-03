

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/xc/cxc3o4kmfernkboftqoiftno7377vidnjfvaggs7kikjumigrd2i.py
# Source Nodes: [grid_sample, grid_sample_4], Original ATen: [aten.grid_sampler_2d]
# grid_sample => full_default, full_default_2
# grid_sample_4 => add_117, add_118, add_119, add_120, convert_element_type_359, convert_element_type_360, convert_element_type_361, convert_element_type_364, floor_8, floor_9, ge_32, ge_33, ge_34, ge_37, logical_and_48, logical_and_49, logical_and_50, logical_and_52, logical_and_53, logical_and_54, logical_and_55, logical_and_56, logical_and_58, logical_and_59, lt_34, lt_35, lt_36, lt_39, mul_213, mul_214, mul_215, mul_216, mul_217, mul_218, sub_115, sub_116, sub_117, sub_120, where_100, where_101, where_102, where_103, where_104, where_105, where_106, where_107, where_96, where_97, where_98, where_99
triton_poi_fused_grid_sampler_2d_44 = async_compile.triton('triton_poi_fused_grid_sampler_2d_44', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*i64', 2: '*i64', 3: '*fp32', 4: '*i64', 5: '*i64', 6: '*fp32', 7: '*i64', 8: '*i64', 9: '*fp32', 10: '*i64', 11: '*i64', 12: '*fp32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_44', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]})
@triton.jit
def triton_poi_fused_grid_sampler_2d_44(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 743424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (1 + (2*x0)), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 176.0
    tmp3 = tmp1 * tmp2
    tmp4 = 175.5
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.floor(tmp5)
    tmp7 = 0.0
    tmp8 = tmp6 >= tmp7
    tmp9 = 352.0
    tmp10 = tmp6 < tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp2
    tmp14 = tmp13 + tmp4
    tmp15 = libdevice.floor(tmp14)
    tmp16 = tmp15 >= tmp7
    tmp17 = tmp15 < tmp9
    tmp18 = tmp16 & tmp17
    tmp19 = tmp10 & tmp18
    tmp20 = tmp8 & tmp19
    tmp21 = tmp15.to(tl.int64)
    tmp22 = tl.full([1], 0, tl.int64)
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = tmp6.to(tl.int64)
    tmp25 = tl.where(tmp20, tmp24, tmp22)
    tmp26 = 1.0
    tmp27 = tmp6 + tmp26
    tmp28 = tmp27 - tmp5
    tmp29 = tmp15 + tmp26
    tmp30 = tmp29 - tmp14
    tmp31 = tmp28 * tmp30
    tmp32 = tl.where(tmp20, tmp31, tmp7)
    tmp33 = tmp27 >= tmp7
    tmp34 = tmp27 < tmp9
    tmp35 = tmp34 & tmp18
    tmp36 = tmp33 & tmp35
    tmp37 = tl.where(tmp36, tmp21, tmp22)
    tmp38 = tmp27.to(tl.int64)
    tmp39 = tl.where(tmp36, tmp38, tmp22)
    tmp40 = tmp5 - tmp6
    tmp41 = tmp40 * tmp30
    tmp42 = tl.where(tmp36, tmp41, tmp7)
    tmp43 = tmp29 >= tmp7
    tmp44 = tmp29 < tmp9
    tmp45 = tmp43 & tmp44
    tmp46 = tmp10 & tmp45
    tmp47 = tmp8 & tmp46
    tmp48 = tmp29.to(tl.int64)
    tmp49 = tl.where(tmp47, tmp48, tmp22)
    tmp50 = tl.where(tmp47, tmp24, tmp22)
    tmp51 = tmp14 - tmp15
    tmp52 = tmp28 * tmp51
    tmp53 = tl.where(tmp47, tmp52, tmp7)
    tmp54 = tmp34 & tmp45
    tmp55 = tmp33 & tmp54
    tmp56 = tl.where(tmp55, tmp48, tmp22)
    tmp57 = tl.where(tmp55, tmp38, tmp22)
    tmp58 = tmp40 * tmp51
    tmp59 = tl.where(tmp55, tmp58, tmp7)
    tl.store(out_ptr0 + (x0), tmp23, None)
    tl.store(out_ptr1 + (x0), tmp25, None)
    tl.store(out_ptr2 + (x0), tmp32, None)
    tl.store(out_ptr3 + (x0), tmp37, None)
    tl.store(out_ptr4 + (x0), tmp39, None)
    tl.store(out_ptr5 + (x0), tmp42, None)
    tl.store(out_ptr6 + (x0), tmp49, None)
    tl.store(out_ptr7 + (x0), tmp50, None)
    tl.store(out_ptr8 + (x0), tmp53, None)
    tl.store(out_ptr9 + (x0), tmp56, None)
    tl.store(out_ptr10 + (x0), tmp57, None)
    tl.store(out_ptr11 + (x0), tmp59, None)
''')
