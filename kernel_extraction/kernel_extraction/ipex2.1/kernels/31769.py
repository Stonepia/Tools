

# Original file: ./detectron2_maskrcnn_r_50_c4__66_inference_106.46/detectron2_maskrcnn_r_50_c4__66_inference_106.46_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/2a/c2al6wk3uezpmyx2oguhmebpqmsr7eunqfihzisfhqnxdudkleck.py
# Source Nodes: [ge, getitem_1, grid_sample], Original ATen: [aten.ge, aten.grid_sampler_2d, aten.index]
# ge => ge_8
# getitem_1 => index
# grid_sample => add_10, add_4, add_5, add_6, add_7, add_8, add_9, convert_element_type_10, convert_element_type_3, convert_element_type_7, floor, floor_1, full_default_2, full_default_4, ge, ge_1, ge_2, ge_5, index_2, index_3, index_4, index_5, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_2, logical_and_4, logical_and_5, logical_and_6, logical_and_7, logical_and_8, lt, lt_1, lt_2, lt_5, mul_10, mul_11, mul_12, mul_13, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, sub_11, sub_6, sub_7, sub_8, where_10, where_11, where_2, where_5, where_8, where_9
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

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ge_grid_sampler_2d_index_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_ge_grid_sampler_2d_index_2(in_ptr0, in_ptr1, out_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9018240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 273280)
    tmp0 = tl.load(in_ptr0 + (2*x2), xmask)
    tmp10 = tl.load(in_ptr0 + (1 + (2*x2)), xmask)
    tmp1 = 7.0
    tmp2 = tmp0 * tmp1
    tmp3 = 6.5
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.floor(tmp4)
    tmp6 = 0.0
    tmp7 = tmp5 >= tmp6
    tmp8 = 14.0
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
    tmp22 = tl.where(tmp21 < 0, tmp21 + 14, tmp21)
    # tl.device_assert((0 <= tmp22) & (tmp22 < 14), "index out of bounds: 0 <= tmp22 < 14")
    tmp23 = tmp5.to(tl.int64)
    tmp24 = tl.where(tmp18, tmp23, tmp20)
    tmp25 = tl.where(tmp24 < 0, tmp24 + 14, tmp24)
    # tl.device_assert((0 <= tmp25) & (tmp25 < 14), "index out of bounds: 0 <= tmp25 < 14")
    tmp26 = tl.load(in_ptr1 + (tmp25 + (14*tmp22) + (196*x1)), xmask).to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 1.0
    tmp29 = tmp5 + tmp28
    tmp30 = tmp29 - tmp4
    tmp31 = tmp13 + tmp28
    tmp32 = tmp31 - tmp12
    tmp33 = tmp30 * tmp32
    tmp34 = tl.where(tmp18, tmp33, tmp6)
    tmp35 = tmp29 >= tmp6
    tmp36 = tmp29 < tmp8
    tmp37 = tmp36 & tmp16
    tmp38 = tmp35 & tmp37
    tmp39 = tl.where(tmp38, tmp19, tmp20)
    tmp40 = tl.where(tmp39 < 0, tmp39 + 14, tmp39)
    # tl.device_assert((0 <= tmp40) & (tmp40 < 14), "index out of bounds: 0 <= tmp40 < 14")
    tmp41 = tmp29.to(tl.int64)
    tmp42 = tl.where(tmp38, tmp41, tmp20)
    tmp43 = tl.where(tmp42 < 0, tmp42 + 14, tmp42)
    # tl.device_assert((0 <= tmp43) & (tmp43 < 14), "index out of bounds: 0 <= tmp43 < 14")
    tmp44 = tl.load(in_ptr1 + (tmp43 + (14*tmp40) + (196*x1)), xmask).to(tl.float32)
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp4 - tmp5
    tmp47 = tmp46 * tmp32
    tmp48 = tl.where(tmp38, tmp47, tmp6)
    tmp49 = tmp31 >= tmp6
    tmp50 = tmp31 < tmp8
    tmp51 = tmp49 & tmp50
    tmp52 = tmp9 & tmp51
    tmp53 = tmp7 & tmp52
    tmp54 = tmp31.to(tl.int64)
    tmp55 = tl.where(tmp53, tmp54, tmp20)
    tmp56 = tl.where(tmp55 < 0, tmp55 + 14, tmp55)
    # tl.device_assert((0 <= tmp56) & (tmp56 < 14), "index out of bounds: 0 <= tmp56 < 14")
    tmp57 = tl.where(tmp53, tmp23, tmp20)
    tmp58 = tl.where(tmp57 < 0, tmp57 + 14, tmp57)
    # tl.device_assert((0 <= tmp58) & (tmp58 < 14), "index out of bounds: 0 <= tmp58 < 14")
    tmp59 = tl.load(in_ptr1 + (tmp58 + (14*tmp56) + (196*x1)), xmask).to(tl.float32)
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp12 - tmp13
    tmp62 = tmp30 * tmp61
    tmp63 = tl.where(tmp53, tmp62, tmp6)
    tmp64 = tmp36 & tmp51
    tmp65 = tmp35 & tmp64
    tmp66 = tl.where(tmp65, tmp54, tmp20)
    tmp67 = tl.where(tmp65, tmp41, tmp20)
    tmp68 = tmp46 * tmp61
    tmp69 = tl.where(tmp65, tmp68, tmp6)
    tmp70 = tmp27 * tmp34
    tmp71 = tmp45 * tmp48
    tmp72 = tmp70 + tmp71
    tmp73 = tmp60 * tmp63
    tmp74 = tmp72 + tmp73
    tmp75 = tl.where(tmp66 < 0, tmp66 + 14, tmp66)
    # tl.device_assert((0 <= tmp75) & (tmp75 < 14), "index out of bounds: 0 <= tmp75 < 14")
    tmp76 = tl.where(tmp67 < 0, tmp67 + 14, tmp67)
    # tl.device_assert((0 <= tmp76) & (tmp76 < 14), "index out of bounds: 0 <= tmp76 < 14")
    tmp77 = tl.load(in_ptr1 + (tmp76 + (14*tmp75) + (196*x1)), xmask).to(tl.float32)
    tmp78 = tmp77.to(tl.float32)
    tmp79 = tmp78 * tmp69
    tmp80 = tmp74 + tmp79
    tmp81 = tmp80.to(tl.float32)
    tmp82 = 0.5
    tmp83 = tmp81 >= tmp82
    tl.store(out_ptr8 + (x2), tmp83, xmask)
''')
