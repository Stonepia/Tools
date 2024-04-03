

# Original file: ./MobileBertForMaskedLM__0_backward_282.1/MobileBertForMaskedLM__0_backward_282.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/jy/cjys44iuwdohseuvvr63odg5b7fxdhe5t4qovrdjocpdsdo5ykfq.py
# Source Nodes: [add_106, add_107, add_121, add_122, add_136, add_137, add_151, add_152, add_166, add_167, add_17, add_181, add_182, add_196, add_197, add_211, add_212, add_226, add_227, add_241, add_242, add_256, add_257, add_271, add_272, add_286, add_287, add_301, add_302, add_31, add_316, add_317, add_32, add_331, add_332, add_346, add_46, add_47, add_61, add_62, add_76, add_77, add_91, add_92, l__self___mobilebert_encoder_layer_10_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_11_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_12_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_13_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_14_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_15_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_16_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_17_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_18_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_19_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_1_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_20_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_21_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_22_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_2_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_3_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_4_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_5_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_6_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_7_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_8_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_9_output_bottleneck_dropout, mul_105, mul_113, mul_121, mul_129, mul_137, mul_145, mul_153, mul_161, mul_169, mul_17, mul_177, mul_25, mul_33, mul_41, mul_49, mul_57, mul_65, mul_73, mul_81, mul_89, mul_9, mul_97], Original ATen: [aten.add, aten.clone, aten.mul]
# add_106 => add_106
# add_107 => add_107
# add_121 => add_121
# add_122 => add_122
# add_136 => add_136
# add_137 => add_137
# add_151 => add_151
# add_152 => add_152
# add_166 => add_166
# add_167 => add_167
# add_17 => add_17
# add_181 => add_181
# add_182 => add_182
# add_196 => add_196
# add_197 => add_197
# add_211 => add_211
# add_212 => add_212
# add_226 => add_226
# add_227 => add_227
# add_241 => add_241
# add_242 => add_242
# add_256 => add_256
# add_257 => add_257
# add_271 => add_271
# add_272 => add_272
# add_286 => add_286
# add_287 => add_287
# add_301 => add_301
# add_302 => add_302
# add_31 => add_31
# add_316 => add_316
# add_317 => add_317
# add_32 => add_32
# add_331 => add_331
# add_332 => add_332
# add_346 => add_346
# add_46 => add_46
# add_47 => add_47
# add_61 => add_61
# add_62 => add_62
# add_76 => add_76
# add_77 => add_77
# add_91 => add_91
# add_92 => add_92
# l__self___mobilebert_encoder_layer_10_output_bottleneck_dropout => clone_55
# l__self___mobilebert_encoder_layer_11_output_bottleneck_dropout => clone_60
# l__self___mobilebert_encoder_layer_12_output_bottleneck_dropout => clone_65
# l__self___mobilebert_encoder_layer_13_output_bottleneck_dropout => clone_70
# l__self___mobilebert_encoder_layer_14_output_bottleneck_dropout => clone_75
# l__self___mobilebert_encoder_layer_15_output_bottleneck_dropout => clone_80
# l__self___mobilebert_encoder_layer_16_output_bottleneck_dropout => clone_85
# l__self___mobilebert_encoder_layer_17_output_bottleneck_dropout => clone_90
# l__self___mobilebert_encoder_layer_18_output_bottleneck_dropout => clone_95
# l__self___mobilebert_encoder_layer_19_output_bottleneck_dropout => clone_100
# l__self___mobilebert_encoder_layer_1_output_bottleneck_dropout => clone_10
# l__self___mobilebert_encoder_layer_20_output_bottleneck_dropout => clone_105
# l__self___mobilebert_encoder_layer_21_output_bottleneck_dropout => clone_110
# l__self___mobilebert_encoder_layer_22_output_bottleneck_dropout => clone_115
# l__self___mobilebert_encoder_layer_2_output_bottleneck_dropout => clone_15
# l__self___mobilebert_encoder_layer_3_output_bottleneck_dropout => clone_20
# l__self___mobilebert_encoder_layer_4_output_bottleneck_dropout => clone_25
# l__self___mobilebert_encoder_layer_5_output_bottleneck_dropout => clone_30
# l__self___mobilebert_encoder_layer_6_output_bottleneck_dropout => clone_35
# l__self___mobilebert_encoder_layer_7_output_bottleneck_dropout => clone_40
# l__self___mobilebert_encoder_layer_8_output_bottleneck_dropout => clone_45
# l__self___mobilebert_encoder_layer_9_output_bottleneck_dropout => clone_50
# mul_105 => mul_131
# mul_113 => mul_141
# mul_121 => mul_151
# mul_129 => mul_161
# mul_137 => mul_171
# mul_145 => mul_181
# mul_153 => mul_191
# mul_161 => mul_201
# mul_169 => mul_211
# mul_17 => mul_21
# mul_177 => mul_221
# mul_25 => mul_31
# mul_33 => mul_41
# mul_41 => mul_51
# mul_49 => mul_61
# mul_57 => mul_71
# mul_65 => mul_81
# mul_73 => mul_91
# mul_81 => mul_101
# mul_89 => mul_111
# mul_9 => mul_11
# mul_97 => mul_121
triton_poi_fused_add_clone_mul_1 = async_compile.triton('triton_poi_fused_add_clone_mul_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*bf16', 8: '*bf16', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*bf16', 14: '*bf16', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*bf16', 20: '*bf16', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*bf16', 26: '*bf16', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*bf16', 32: '*bf16', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*bf16', 38: '*bf16', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*bf16', 44: '*bf16', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*bf16', 50: '*bf16', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*bf16', 56: '*bf16', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*bf16', 62: '*bf16', 63: '*fp32', 64: '*fp32', 65: '*fp32', 66: '*fp32', 67: '*fp32', 68: '*fp32', 69: '*fp32', 70: '*fp32', 71: '*fp32', 72: '*fp32', 73: '*fp32', 74: '*fp32', 75: '*fp32', 76: '*fp32', 77: '*fp32', 78: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_mul_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_mul_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x2), None).to(tl.float32)
    tmp17 = tl.load(in_ptr8 + (x2), None).to(tl.float32)
    tmp19 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr13 + (x2), None).to(tl.float32)
    tmp31 = tl.load(in_ptr14 + (x2), None).to(tl.float32)
    tmp33 = tl.load(in_ptr15 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr16 + (x0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr17 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr18 + (x0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr19 + (x2), None).to(tl.float32)
    tmp45 = tl.load(in_ptr20 + (x2), None).to(tl.float32)
    tmp47 = tl.load(in_ptr21 + (x0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr22 + (x0), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr23 + (x0), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr24 + (x0), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr25 + (x2), None).to(tl.float32)
    tmp59 = tl.load(in_ptr26 + (x2), None).to(tl.float32)
    tmp61 = tl.load(in_ptr27 + (x0), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr28 + (x0), None, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr29 + (x0), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr30 + (x0), None, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr31 + (x2), None).to(tl.float32)
    tmp73 = tl.load(in_ptr32 + (x2), None).to(tl.float32)
    tmp75 = tl.load(in_ptr33 + (x0), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr34 + (x0), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr35 + (x0), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr36 + (x0), None, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr37 + (x2), None).to(tl.float32)
    tmp87 = tl.load(in_ptr38 + (x2), None).to(tl.float32)
    tmp89 = tl.load(in_ptr39 + (x0), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr40 + (x0), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr41 + (x0), None, eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr42 + (x0), None, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr43 + (x2), None).to(tl.float32)
    tmp101 = tl.load(in_ptr44 + (x2), None).to(tl.float32)
    tmp103 = tl.load(in_ptr45 + (x0), None, eviction_policy='evict_last')
    tmp105 = tl.load(in_ptr46 + (x0), None, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr47 + (x0), None, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr48 + (x0), None, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr49 + (x2), None).to(tl.float32)
    tmp115 = tl.load(in_ptr50 + (x2), None).to(tl.float32)
    tmp117 = tl.load(in_ptr51 + (x0), None, eviction_policy='evict_last')
    tmp119 = tl.load(in_ptr52 + (x0), None, eviction_policy='evict_last')
    tmp122 = tl.load(in_ptr53 + (x0), None, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr54 + (x0), None, eviction_policy='evict_last')
    tmp127 = tl.load(in_ptr55 + (x2), None).to(tl.float32)
    tmp129 = tl.load(in_ptr56 + (x2), None).to(tl.float32)
    tmp131 = tl.load(in_ptr57 + (x0), None, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr58 + (x0), None, eviction_policy='evict_last')
    tmp136 = tl.load(in_ptr59 + (x0), None, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr60 + (x0), None, eviction_policy='evict_last')
    tmp141 = tl.load(in_ptr61 + (x2), None).to(tl.float32)
    tmp143 = tl.load(in_ptr62 + (x2), None).to(tl.float32)
    tmp145 = tl.load(in_ptr63 + (x0), None, eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr64 + (x0), None, eviction_policy='evict_last')
    tmp150 = tl.load(in_ptr65 + (x0), None, eviction_policy='evict_last')
    tmp152 = tl.load(in_ptr66 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 + tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tmp1 + tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tmp18 + tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp16 + tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = tmp32 + tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tmp30 + tmp41
    tmp44 = tmp43.to(tl.float32)
    tmp46 = tmp45.to(tl.float32)
    tmp48 = tmp42 * tmp47
    tmp50 = tmp48 + tmp49
    tmp51 = tmp46 + tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tmp44 + tmp55
    tmp58 = tmp57.to(tl.float32)
    tmp60 = tmp59.to(tl.float32)
    tmp62 = tmp56 * tmp61
    tmp64 = tmp62 + tmp63
    tmp65 = tmp60 + tmp64
    tmp67 = tmp65 * tmp66
    tmp69 = tmp67 + tmp68
    tmp70 = tmp58 + tmp69
    tmp72 = tmp71.to(tl.float32)
    tmp74 = tmp73.to(tl.float32)
    tmp76 = tmp70 * tmp75
    tmp78 = tmp76 + tmp77
    tmp79 = tmp74 + tmp78
    tmp81 = tmp79 * tmp80
    tmp83 = tmp81 + tmp82
    tmp84 = tmp72 + tmp83
    tmp86 = tmp85.to(tl.float32)
    tmp88 = tmp87.to(tl.float32)
    tmp90 = tmp84 * tmp89
    tmp92 = tmp90 + tmp91
    tmp93 = tmp88 + tmp92
    tmp95 = tmp93 * tmp94
    tmp97 = tmp95 + tmp96
    tmp98 = tmp86 + tmp97
    tmp100 = tmp99.to(tl.float32)
    tmp102 = tmp101.to(tl.float32)
    tmp104 = tmp98 * tmp103
    tmp106 = tmp104 + tmp105
    tmp107 = tmp102 + tmp106
    tmp109 = tmp107 * tmp108
    tmp111 = tmp109 + tmp110
    tmp112 = tmp100 + tmp111
    tmp114 = tmp113.to(tl.float32)
    tmp116 = tmp115.to(tl.float32)
    tmp118 = tmp112 * tmp117
    tmp120 = tmp118 + tmp119
    tmp121 = tmp116 + tmp120
    tmp123 = tmp121 * tmp122
    tmp125 = tmp123 + tmp124
    tmp126 = tmp114 + tmp125
    tmp128 = tmp127.to(tl.float32)
    tmp130 = tmp129.to(tl.float32)
    tmp132 = tmp126 * tmp131
    tmp134 = tmp132 + tmp133
    tmp135 = tmp130 + tmp134
    tmp137 = tmp135 * tmp136
    tmp139 = tmp137 + tmp138
    tmp140 = tmp128 + tmp139
    tmp142 = tmp141.to(tl.float32)
    tmp144 = tmp143.to(tl.float32)
    tmp146 = tmp140 * tmp145
    tmp148 = tmp146 + tmp147
    tmp149 = tmp144 + tmp148
    tmp151 = tmp149 * tmp150
    tmp153 = tmp151 + tmp152
    tmp154 = tmp142 + tmp153
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp28, None)
    tl.store(out_ptr2 + (x2), tmp42, None)
    tl.store(out_ptr3 + (x2), tmp56, None)
    tl.store(out_ptr4 + (x2), tmp70, None)
    tl.store(out_ptr5 + (x2), tmp84, None)
    tl.store(out_ptr6 + (x2), tmp98, None)
    tl.store(out_ptr7 + (x2), tmp112, None)
    tl.store(out_ptr8 + (x2), tmp126, None)
    tl.store(out_ptr9 + (x2), tmp140, None)
    tl.store(out_ptr10 + (x2), tmp154, None)
''')
