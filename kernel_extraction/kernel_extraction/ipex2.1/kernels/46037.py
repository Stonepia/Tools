

# Original file: ./detectron2_maskrcnn_r_101_fpn__79_inference_119.59/detectron2_maskrcnn_r_101_fpn__79_inference_119.59_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ee/cee6my5433e3qtazrnsrkt27xh6txdgoyzmlwnoqiomoups42oio.py
# Source Nodes: [ge, getitem_1, grid_sample], Original ATen: [aten.ge, aten.grid_sampler_2d, aten.index]
# ge => ge_8
# getitem_1 => index
# grid_sample => add_10, add_4, add_5, add_6, add_7, add_8, add_9, convert_element_type_4, convert_element_type_7, floor, floor_1, full_default_2, full_default_4, ge, ge_1, ge_2, ge_5, index_2, index_3, index_4, index_5, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_2, logical_and_4, logical_and_5, logical_and_6, logical_and_7, logical_and_8, lt, lt_1, lt_2, lt_5, mul_10, mul_11, mul_12, mul_13, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, sub_11, sub_6, sub_7, sub_8, where_10, where_11, where_2, where_5, where_8, where_9
triton_poi_fused_ge_grid_sampler_2d_index_2 = async_compile.triton('triton_poi_fused_ge_grid_sampler_2d_index_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ge_grid_sampler_2d_index_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_ge_grid_sampler_2d_index_2(in_ptr0, in_ptr1, out_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10931200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 273280)
    tmp0 = tl.load(in_ptr0 + (2*x2), xmask)
    tmp10 = tl.load(in_ptr0 + (1 + (2*x2)), xmask)
    tmp1 = 14.0
    tmp2 = tmp0 * tmp1
    tmp3 = 13.5
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.floor(tmp4)
    tmp6 = 0.0
    tmp7 = tmp5 >= tmp6
    tmp8 = 28.0
    tmp9 = tmp5 < tmp8
    tmp11 = tmp10 * tmp1
    tmp12 = tmp11 + tmp3
    tmp13 = libdevice.floor(tmp12)
    tmp14 = tmp13 >= tmp6
    tmp15 = tmp13 < tmp8
    tmp16 = tmp14 & tmp15
    tmp17 = tmp9 & tmp16
    tmp18 = tmp7 & tmp17
    tmp19 = tmp13.to(tl.int64)
    tmp20 = tl.full([1], 0, tl.int64)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp21 < 0, tmp21 + 28, tmp21)
    # tl.device_assert((0 <= tmp22) & (tmp22 < 28), "index out of bounds: 0 <= tmp22 < 28")
    tmp23 = tmp5.to(tl.int64)
    tmp24 = tl.where(tmp18, tmp23, tmp20)
    tmp25 = tl.where(tmp24 < 0, tmp24 + 28, tmp24)
    # tl.device_assert((0 <= tmp25) & (tmp25 < 28), "index out of bounds: 0 <= tmp25 < 28")
    tmp26 = tl.load(in_ptr1 + (tmp25 + (28*tmp22) + (784*x1)), xmask)
    tmp27 = 1.0
    tmp28 = tmp5 + tmp27
    tmp29 = tmp28 - tmp4
    tmp30 = tmp13 + tmp27
    tmp31 = tmp30 - tmp12
    tmp32 = tmp29 * tmp31
    tmp33 = tl.where(tmp18, tmp32, tmp6)
    tmp34 = tmp28 >= tmp6
    tmp35 = tmp28 < tmp8
    tmp36 = tmp35 & tmp16
    tmp37 = tmp34 & tmp36
    tmp38 = tl.where(tmp37, tmp19, tmp20)
    tmp39 = tl.where(tmp38 < 0, tmp38 + 28, tmp38)
    # tl.device_assert((0 <= tmp39) & (tmp39 < 28), "index out of bounds: 0 <= tmp39 < 28")
    tmp40 = tmp28.to(tl.int64)
    tmp41 = tl.where(tmp37, tmp40, tmp20)
    tmp42 = tl.where(tmp41 < 0, tmp41 + 28, tmp41)
    # tl.device_assert((0 <= tmp42) & (tmp42 < 28), "index out of bounds: 0 <= tmp42 < 28")
    tmp43 = tl.load(in_ptr1 + (tmp42 + (28*tmp39) + (784*x1)), xmask)
    tmp44 = tmp4 - tmp5
    tmp45 = tmp44 * tmp31
    tmp46 = tl.where(tmp37, tmp45, tmp6)
    tmp47 = tmp30 >= tmp6
    tmp48 = tmp30 < tmp8
    tmp49 = tmp47 & tmp48
    tmp50 = tmp9 & tmp49
    tmp51 = tmp7 & tmp50
    tmp52 = tmp30.to(tl.int64)
    tmp53 = tl.where(tmp51, tmp52, tmp20)
    tmp54 = tl.where(tmp53 < 0, tmp53 + 28, tmp53)
    # tl.device_assert((0 <= tmp54) & (tmp54 < 28), "index out of bounds: 0 <= tmp54 < 28")
    tmp55 = tl.where(tmp51, tmp23, tmp20)
    tmp56 = tl.where(tmp55 < 0, tmp55 + 28, tmp55)
    # tl.device_assert((0 <= tmp56) & (tmp56 < 28), "index out of bounds: 0 <= tmp56 < 28")
    tmp57 = tl.load(in_ptr1 + (tmp56 + (28*tmp54) + (784*x1)), xmask)
    tmp58 = tmp12 - tmp13
    tmp59 = tmp29 * tmp58
    tmp60 = tl.where(tmp51, tmp59, tmp6)
    tmp61 = tmp35 & tmp49
    tmp62 = tmp34 & tmp61
    tmp63 = tl.where(tmp62, tmp52, tmp20)
    tmp64 = tl.where(tmp62, tmp40, tmp20)
    tmp65 = tmp44 * tmp58
    tmp66 = tl.where(tmp62, tmp65, tmp6)
    tmp67 = tmp26 * tmp33
    tmp68 = tmp43 * tmp46
    tmp69 = tmp67 + tmp68
    tmp70 = tmp57 * tmp60
    tmp71 = tmp69 + tmp70
    tmp72 = tl.where(tmp63 < 0, tmp63 + 28, tmp63)
    # tl.device_assert((0 <= tmp72) & (tmp72 < 28), "index out of bounds: 0 <= tmp72 < 28")
    tmp73 = tl.where(tmp64 < 0, tmp64 + 28, tmp64)
    # tl.device_assert((0 <= tmp73) & (tmp73 < 28), "index out of bounds: 0 <= tmp73 < 28")
    tmp74 = tl.load(in_ptr1 + (tmp73 + (28*tmp72) + (784*x1)), xmask)
    tmp75 = tmp74 * tmp66
    tmp76 = tmp71 + tmp75
    tmp77 = 0.5
    tmp78 = tmp76 >= tmp77
    tl.store(out_ptr8 + (x2), tmp78, xmask)
''')
