

# Original file: ./MobileBertForQuestionAnswering__0_backward_207.1/MobileBertForQuestionAnswering__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/6e/c6e4r3xwsmhsxwhj7rux2yjgc7ebcheadagir5rr7bnvlbca774i.py
# Source Nodes: [add_106, add_107, add_121, add_122, add_136, add_137, add_151, add_152, add_166, add_167, add_17, add_181, add_182, add_196, add_197, add_211, add_212, add_226, add_227, add_241, add_242, add_256, add_257, add_271, add_272, add_286, add_287, add_301, add_302, add_31, add_316, add_317, add_32, add_331, add_332, add_346, add_46, add_47, add_61, add_62, add_76, add_77, add_91, add_92, l__mod___mobilebert_encoder_layer_10_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_11_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_12_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_13_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_14_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_15_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_16_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_17_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_18_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_19_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_1_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_20_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_21_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_22_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_2_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_3_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_4_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_5_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_6_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_7_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_8_output_bottleneck_dropout, l__mod___mobilebert_encoder_layer_9_output_bottleneck_dropout, mul_105, mul_113, mul_121, mul_129, mul_137, mul_145, mul_153, mul_161, mul_169, mul_17, mul_177, mul_25, mul_33, mul_41, mul_49, mul_57, mul_65, mul_73, mul_81, mul_89, mul_9, mul_97], Original ATen: [aten.add, aten.clone, aten.mul]
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
# l__mod___mobilebert_encoder_layer_10_output_bottleneck_dropout => clone_55
# l__mod___mobilebert_encoder_layer_11_output_bottleneck_dropout => clone_60
# l__mod___mobilebert_encoder_layer_12_output_bottleneck_dropout => clone_65
# l__mod___mobilebert_encoder_layer_13_output_bottleneck_dropout => clone_70
# l__mod___mobilebert_encoder_layer_14_output_bottleneck_dropout => clone_75
# l__mod___mobilebert_encoder_layer_15_output_bottleneck_dropout => clone_80
# l__mod___mobilebert_encoder_layer_16_output_bottleneck_dropout => clone_85
# l__mod___mobilebert_encoder_layer_17_output_bottleneck_dropout => clone_90
# l__mod___mobilebert_encoder_layer_18_output_bottleneck_dropout => clone_95
# l__mod___mobilebert_encoder_layer_19_output_bottleneck_dropout => clone_100
# l__mod___mobilebert_encoder_layer_1_output_bottleneck_dropout => clone_10
# l__mod___mobilebert_encoder_layer_20_output_bottleneck_dropout => clone_105
# l__mod___mobilebert_encoder_layer_21_output_bottleneck_dropout => clone_110
# l__mod___mobilebert_encoder_layer_22_output_bottleneck_dropout => clone_115
# l__mod___mobilebert_encoder_layer_2_output_bottleneck_dropout => clone_15
# l__mod___mobilebert_encoder_layer_3_output_bottleneck_dropout => clone_20
# l__mod___mobilebert_encoder_layer_4_output_bottleneck_dropout => clone_25
# l__mod___mobilebert_encoder_layer_5_output_bottleneck_dropout => clone_30
# l__mod___mobilebert_encoder_layer_6_output_bottleneck_dropout => clone_35
# l__mod___mobilebert_encoder_layer_7_output_bottleneck_dropout => clone_40
# l__mod___mobilebert_encoder_layer_8_output_bottleneck_dropout => clone_45
# l__mod___mobilebert_encoder_layer_9_output_bottleneck_dropout => clone_50
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

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*bf16', 12: '*bf16', 13: '*bf16', 14: '*bf16', 15: '*bf16', 16: '*bf16', 17: '*bf16', 18: '*bf16', 19: '*bf16', 20: '*bf16', 21: '*bf16', 22: '*bf16', 23: '*bf16', 24: '*bf16', 25: '*bf16', 26: '*bf16', 27: '*bf16', 28: '*bf16', 29: '*bf16', 30: '*bf16', 31: '*bf16', 32: '*bf16', 33: '*bf16', 34: '*bf16', 35: '*bf16', 36: '*bf16', 37: '*bf16', 38: '*bf16', 39: '*bf16', 40: '*bf16', 41: '*bf16', 42: '*bf16', 43: '*bf16', 44: '*bf16', 45: '*bf16', 46: '*bf16', 47: '*bf16', 48: '*bf16', 49: '*bf16', 50: '*bf16', 51: '*bf16', 52: '*bf16', 53: '*bf16', 54: '*bf16', 55: '*bf16', 56: '*bf16', 57: '*bf16', 58: '*bf16', 59: '*bf16', 60: '*bf16', 61: '*bf16', 62: '*bf16', 63: '*bf16', 64: '*bf16', 65: '*bf16', 66: '*bf16', 67: '*bf16', 68: '*bf16', 69: '*bf16', 70: '*bf16', 71: '*bf16', 72: '*bf16', 73: '*bf16', 74: '*bf16', 75: '*bf16', 76: '*bf16', 77: '*bf16', 78: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_mul_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_mul_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr7 + (x2), None).to(tl.float32)
    tmp14 = tl.load(in_ptr8 + (x2), None).to(tl.float32)
    tmp15 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tl.load(in_ptr13 + (x2), None).to(tl.float32)
    tmp26 = tl.load(in_ptr14 + (x2), None).to(tl.float32)
    tmp27 = tl.load(in_ptr15 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp29 = tl.load(in_ptr16 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp32 = tl.load(in_ptr17 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp34 = tl.load(in_ptr18 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp37 = tl.load(in_ptr19 + (x2), None).to(tl.float32)
    tmp38 = tl.load(in_ptr20 + (x2), None).to(tl.float32)
    tmp39 = tl.load(in_ptr21 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp41 = tl.load(in_ptr22 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp44 = tl.load(in_ptr23 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp46 = tl.load(in_ptr24 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp49 = tl.load(in_ptr25 + (x2), None).to(tl.float32)
    tmp50 = tl.load(in_ptr26 + (x2), None).to(tl.float32)
    tmp51 = tl.load(in_ptr27 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp53 = tl.load(in_ptr28 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp56 = tl.load(in_ptr29 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp58 = tl.load(in_ptr30 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp61 = tl.load(in_ptr31 + (x2), None).to(tl.float32)
    tmp62 = tl.load(in_ptr32 + (x2), None).to(tl.float32)
    tmp63 = tl.load(in_ptr33 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp65 = tl.load(in_ptr34 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp68 = tl.load(in_ptr35 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp70 = tl.load(in_ptr36 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp73 = tl.load(in_ptr37 + (x2), None).to(tl.float32)
    tmp74 = tl.load(in_ptr38 + (x2), None).to(tl.float32)
    tmp75 = tl.load(in_ptr39 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp77 = tl.load(in_ptr40 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp80 = tl.load(in_ptr41 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp82 = tl.load(in_ptr42 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp85 = tl.load(in_ptr43 + (x2), None).to(tl.float32)
    tmp86 = tl.load(in_ptr44 + (x2), None).to(tl.float32)
    tmp87 = tl.load(in_ptr45 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp89 = tl.load(in_ptr46 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp92 = tl.load(in_ptr47 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp94 = tl.load(in_ptr48 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp97 = tl.load(in_ptr49 + (x2), None).to(tl.float32)
    tmp98 = tl.load(in_ptr50 + (x2), None).to(tl.float32)
    tmp99 = tl.load(in_ptr51 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp101 = tl.load(in_ptr52 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp104 = tl.load(in_ptr53 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp106 = tl.load(in_ptr54 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp109 = tl.load(in_ptr55 + (x2), None).to(tl.float32)
    tmp110 = tl.load(in_ptr56 + (x2), None).to(tl.float32)
    tmp111 = tl.load(in_ptr57 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp113 = tl.load(in_ptr58 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp116 = tl.load(in_ptr59 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp118 = tl.load(in_ptr60 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp121 = tl.load(in_ptr61 + (x2), None).to(tl.float32)
    tmp122 = tl.load(in_ptr62 + (x2), None).to(tl.float32)
    tmp123 = tl.load(in_ptr63 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp125 = tl.load(in_ptr64 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp128 = tl.load(in_ptr65 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp130 = tl.load(in_ptr66 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp1 + tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp0 + tmp11
    tmp16 = tmp12 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp14 + tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tmp13 + tmp23
    tmp28 = tmp24 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp26 + tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp25 + tmp35
    tmp40 = tmp36 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp38 + tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = tmp37 + tmp47
    tmp52 = tmp48 * tmp51
    tmp54 = tmp52 + tmp53
    tmp55 = tmp50 + tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tmp49 + tmp59
    tmp64 = tmp60 * tmp63
    tmp66 = tmp64 + tmp65
    tmp67 = tmp62 + tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = tmp61 + tmp71
    tmp76 = tmp72 * tmp75
    tmp78 = tmp76 + tmp77
    tmp79 = tmp74 + tmp78
    tmp81 = tmp79 * tmp80
    tmp83 = tmp81 + tmp82
    tmp84 = tmp73 + tmp83
    tmp88 = tmp84 * tmp87
    tmp90 = tmp88 + tmp89
    tmp91 = tmp86 + tmp90
    tmp93 = tmp91 * tmp92
    tmp95 = tmp93 + tmp94
    tmp96 = tmp85 + tmp95
    tmp100 = tmp96 * tmp99
    tmp102 = tmp100 + tmp101
    tmp103 = tmp98 + tmp102
    tmp105 = tmp103 * tmp104
    tmp107 = tmp105 + tmp106
    tmp108 = tmp97 + tmp107
    tmp112 = tmp108 * tmp111
    tmp114 = tmp112 + tmp113
    tmp115 = tmp110 + tmp114
    tmp117 = tmp115 * tmp116
    tmp119 = tmp117 + tmp118
    tmp120 = tmp109 + tmp119
    tmp124 = tmp120 * tmp123
    tmp126 = tmp124 + tmp125
    tmp127 = tmp122 + tmp126
    tmp129 = tmp127 * tmp128
    tmp131 = tmp129 + tmp130
    tmp132 = tmp121 + tmp131
    tl.store(out_ptr0 + (x2), tmp12, None)
    tl.store(out_ptr1 + (x2), tmp24, None)
    tl.store(out_ptr2 + (x2), tmp36, None)
    tl.store(out_ptr3 + (x2), tmp48, None)
    tl.store(out_ptr4 + (x2), tmp60, None)
    tl.store(out_ptr5 + (x2), tmp72, None)
    tl.store(out_ptr6 + (x2), tmp84, None)
    tl.store(out_ptr7 + (x2), tmp96, None)
    tl.store(out_ptr8 + (x2), tmp108, None)
    tl.store(out_ptr9 + (x2), tmp120, None)
    tl.store(out_ptr10 + (x2), tmp132, None)
''')
