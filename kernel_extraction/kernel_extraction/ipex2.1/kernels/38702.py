

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/6g/c6gtbw3hk7lox5nw5l3o7c3su4ie4lwr4wdhdkb5xtwimdm2icdm.py
# Source Nodes: [add_12, grid_sample_2, grid_sample_3, l__mod___vgg16_conv_4_3_1, mul_15, mul_16, mul_17, mul_18, sigmoid, sub_7, truediv_8], Original ATen: [aten.add, aten.div, aten.grid_sampler_2d, aten.mul, aten.relu, aten.rsub, aten.sigmoid]
# add_12 => add_112
# grid_sample_2 => add_100, add_101, add_99, index_14, index_15, index_16, index_17, mul_187, mul_188, mul_189, mul_190
# grid_sample_3 => add_108, add_109, add_110, index_18, index_19, index_20, index_21, mul_199, mul_200, mul_201, mul_202
# l__mod___vgg16_conv_4_3_1 => fn_18
# mul_15 => mul_205
# mul_16 => mul_206
# mul_17 => mul_207
# mul_18 => mul_208
# sigmoid => sigmoid
# sub_7 => sub_85
# truediv_8 => div_8
triton_poi_fused_add_div_grid_sampler_2d_mul_relu_rsub_sigmoid_29 = async_compile.triton('triton_poi_fused_add_div_grid_sampler_2d_mul_relu_rsub_sigmoid_29', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32, 131072], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*i64', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*i64', 15: '*i64', 16: '*fp32', 17: '*i64', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: 'i32', 22: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_grid_sampler_2d_mul_relu_rsub_sigmoid_29', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_grid_sampler_2d_mul_relu_rsub_sigmoid_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18
    xnumel = 123904
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = (yindex // 3)
    y3 = yindex
    y0 = yindex % 3
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (247808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + (2*x2) + (247808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr2 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr3 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr4 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr5 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr6 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr7 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr8 + ((2*x2) + (247808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr8 + (1 + (2*x2) + (247808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr10 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr11 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp117 = tl.load(in_ptr12 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp120 = tl.load(in_ptr13 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp122 = tl.load(in_ptr14 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp125 = tl.load(in_ptr15 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp128 = tl.load(in_ptr16 + (y1), ymask, eviction_policy='evict_last')
    tmp144 = tl.load(in_ptr17 + (4 + (5*x2) + (619520*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp152 = tl.load(in_ptr18 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tmp19 = tmp13.to(tl.int64)
    tmp20 = tl.full([1, 1], 0, tl.int64)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp21 < 0, tmp21 + 352, tmp21)
    # tl.device_assert((0 <= tmp22) & (tmp22 < 352), "index out of bounds: 0 <= tmp22 < 352")
    tmp23 = tmp5.to(tl.int64)
    tmp24 = tl.where(tmp18, tmp23, tmp20)
    tmp25 = tl.where(tmp24 < 0, tmp24 + 352, tmp24)
    # tl.device_assert((0 <= tmp25) & (tmp25 < 352), "index out of bounds: 0 <= tmp25 < 352")
    tmp26 = tl.load(in_ptr1 + (tmp25 + (352*tmp22) + (123904*y3)), ymask)
    tmp27 = 1.0
    tmp28 = tmp5 + tmp27
    tmp29 = tmp28 >= tmp6
    tmp30 = tmp28 < tmp8
    tmp31 = tmp30 & tmp16
    tmp32 = tmp29 & tmp31
    tmp33 = tl.where(tmp32, tmp19, tmp20)
    tmp34 = tl.where(tmp33 < 0, tmp33 + 352, tmp33)
    # tl.device_assert((0 <= tmp34) & (tmp34 < 352), "index out of bounds: 0 <= tmp34 < 352")
    tmp35 = tmp28.to(tl.int64)
    tmp36 = tl.where(tmp32, tmp35, tmp20)
    tmp37 = tl.where(tmp36 < 0, tmp36 + 352, tmp36)
    # tl.device_assert((0 <= tmp37) & (tmp37 < 352), "index out of bounds: 0 <= tmp37 < 352")
    tmp38 = tl.load(in_ptr1 + (tmp37 + (352*tmp34) + (123904*y3)), ymask)
    tmp39 = tmp13 + tmp27
    tmp40 = tmp39 >= tmp6
    tmp41 = tmp39 < tmp8
    tmp42 = tmp40 & tmp41
    tmp43 = tmp9 & tmp42
    tmp44 = tmp7 & tmp43
    tmp45 = tmp39.to(tl.int64)
    tmp46 = tl.where(tmp44, tmp45, tmp20)
    tmp47 = tl.where(tmp46 < 0, tmp46 + 352, tmp46)
    # tl.device_assert((0 <= tmp47) & (tmp47 < 352), "index out of bounds: 0 <= tmp47 < 352")
    tmp48 = tl.where(tmp44, tmp23, tmp20)
    tmp49 = tl.where(tmp48 < 0, tmp48 + 352, tmp48)
    # tl.device_assert((0 <= tmp49) & (tmp49 < 352), "index out of bounds: 0 <= tmp49 < 352")
    tmp50 = tl.load(in_ptr1 + (tmp49 + (352*tmp47) + (123904*y3)), ymask)
    tmp52 = tmp26 * tmp51
    tmp54 = tmp38 * tmp53
    tmp55 = tmp52 + tmp54
    tmp57 = tmp50 * tmp56
    tmp58 = tmp55 + tmp57
    tmp60 = tl.where(tmp59 < 0, tmp59 + 352, tmp59)
    # tl.device_assert(((0 <= tmp60) & (tmp60 < 352)) | ~(ymask & xmask), "index out of bounds: 0 <= tmp60 < 352")
    tmp62 = tl.where(tmp61 < 0, tmp61 + 352, tmp61)
    # tl.device_assert(((0 <= tmp62) & (tmp62 < 352)) | ~(ymask & xmask), "index out of bounds: 0 <= tmp62 < 352")
    tmp63 = tl.load(in_ptr1 + (tmp62 + (352*tmp60) + (123904*y3)), xmask & ymask)
    tmp65 = tmp63 * tmp64
    tmp66 = tmp58 + tmp65
    tmp68 = tmp67 * tmp1
    tmp69 = tmp68 + tmp3
    tmp70 = libdevice.floor(tmp69)
    tmp71 = tmp70 >= tmp6
    tmp72 = tmp70 < tmp8
    tmp74 = tmp73 * tmp1
    tmp75 = tmp74 + tmp3
    tmp76 = libdevice.floor(tmp75)
    tmp77 = tmp76 >= tmp6
    tmp78 = tmp76 < tmp8
    tmp79 = tmp77 & tmp78
    tmp80 = tmp72 & tmp79
    tmp81 = tmp71 & tmp80
    tmp82 = tmp76.to(tl.int64)
    tmp83 = tl.where(tmp81, tmp82, tmp20)
    tmp84 = tl.where(tmp83 < 0, tmp83 + 352, tmp83)
    # tl.device_assert((0 <= tmp84) & (tmp84 < 352), "index out of bounds: 0 <= tmp84 < 352")
    tmp85 = tmp70.to(tl.int64)
    tmp86 = tl.where(tmp81, tmp85, tmp20)
    tmp87 = tl.where(tmp86 < 0, tmp86 + 352, tmp86)
    # tl.device_assert((0 <= tmp87) & (tmp87 < 352), "index out of bounds: 0 <= tmp87 < 352")
    tmp88 = tl.load(in_ptr9 + (tmp87 + (352*tmp84) + (123904*y3)), ymask)
    tmp89 = tmp70 + tmp27
    tmp90 = tmp89 >= tmp6
    tmp91 = tmp89 < tmp8
    tmp92 = tmp91 & tmp79
    tmp93 = tmp90 & tmp92
    tmp94 = tl.where(tmp93, tmp82, tmp20)
    tmp95 = tl.where(tmp94 < 0, tmp94 + 352, tmp94)
    # tl.device_assert((0 <= tmp95) & (tmp95 < 352), "index out of bounds: 0 <= tmp95 < 352")
    tmp96 = tmp89.to(tl.int64)
    tmp97 = tl.where(tmp93, tmp96, tmp20)
    tmp98 = tl.where(tmp97 < 0, tmp97 + 352, tmp97)
    # tl.device_assert((0 <= tmp98) & (tmp98 < 352), "index out of bounds: 0 <= tmp98 < 352")
    tmp99 = tl.load(in_ptr9 + (tmp98 + (352*tmp95) + (123904*y3)), ymask)
    tmp100 = tmp76 + tmp27
    tmp101 = tmp100 >= tmp6
    tmp102 = tmp100 < tmp8
    tmp103 = tmp101 & tmp102
    tmp104 = tmp72 & tmp103
    tmp105 = tmp71 & tmp104
    tmp106 = tmp100.to(tl.int64)
    tmp107 = tl.where(tmp105, tmp106, tmp20)
    tmp108 = tl.where(tmp107 < 0, tmp107 + 352, tmp107)
    # tl.device_assert((0 <= tmp108) & (tmp108 < 352), "index out of bounds: 0 <= tmp108 < 352")
    tmp109 = tl.where(tmp105, tmp85, tmp20)
    tmp110 = tl.where(tmp109 < 0, tmp109 + 352, tmp109)
    # tl.device_assert((0 <= tmp110) & (tmp110 < 352), "index out of bounds: 0 <= tmp110 < 352")
    tmp111 = tl.load(in_ptr9 + (tmp110 + (352*tmp108) + (123904*y3)), ymask)
    tmp113 = tmp88 * tmp112
    tmp115 = tmp99 * tmp114
    tmp116 = tmp113 + tmp115
    tmp118 = tmp111 * tmp117
    tmp119 = tmp116 + tmp118
    tmp121 = tl.where(tmp120 < 0, tmp120 + 352, tmp120)
    # tl.device_assert(((0 <= tmp121) & (tmp121 < 352)) | ~(ymask & xmask), "index out of bounds: 0 <= tmp121 < 352")
    tmp123 = tl.where(tmp122 < 0, tmp122 + 352, tmp122)
    # tl.device_assert(((0 <= tmp123) & (tmp123 < 352)) | ~(ymask & xmask), "index out of bounds: 0 <= tmp123 < 352")
    tmp124 = tl.load(in_ptr9 + (tmp123 + (352*tmp121) + (123904*y3)), xmask & ymask)
    tmp126 = tmp124 * tmp125
    tmp127 = tmp119 + tmp126
    tmp129 = tl.where(tmp128 < 0, tmp128 + 7, tmp128)
    # tl.device_assert(((0 <= tmp129) & (tmp129 < 7)) | ~ymask, "index out of bounds: 0 <= tmp129 < 7")
    tmp130 = tmp129
    tmp131 = tmp130.to(tl.float32)
    tmp132 = 3.5
    tmp133 = tmp131 < tmp132
    tmp134 = 0.125
    tmp135 = tmp131 * tmp134
    tmp136 = tmp135 + tmp134
    tmp137 = 6 + ((-1)*tmp129)
    tmp138 = tmp137.to(tl.float32)
    tmp139 = tmp138 * tmp134
    tmp140 = 0.875
    tmp141 = tmp140 - tmp139
    tmp142 = tl.where(tmp133, tmp136, tmp141)
    tmp143 = tmp27 - tmp142
    tmp145 = tl.sigmoid(tmp144)
    tmp146 = tmp143 * tmp145
    tmp147 = tmp146 * tmp66
    tmp148 = tmp27 - tmp145
    tmp149 = tmp142 * tmp148
    tmp150 = tmp149 * tmp127
    tmp151 = tmp147 + tmp150
    tmp153 = tmp151 / tmp152
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (123904*y3)), tmp153, xmask & ymask)
    tl.store(out_ptr4 + (y0 + (3*x2) + (371712*y1)), tmp153, xmask & ymask)
''')
