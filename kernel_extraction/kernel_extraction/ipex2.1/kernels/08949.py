

# Original file: ./MegatronBertForCausalLM__0_backward_207.1/MegatronBertForCausalLM__0_backward_207.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/wy/cwyp4bbeag6zg5rtwou3nkmkii4o3xaimf4zfw7wpoasa7rqlwbe.py
# Source Nodes: [add_11, add_12, add_14, add_15, add_17, add_18, add_2, add_20, add_21, add_23, add_24, add_26, add_27, add_29, add_3, add_30, add_32, add_33, add_35, add_36, add_38, add_39, add_41, add_42, add_44, add_45, add_47, add_48, add_5, add_50, add_51, add_53, add_54, add_56, add_57, add_59, add_6, add_60, add_62, add_63, add_65, add_66, add_68, add_69, add_71, add_72, add_8, add_9, l__self___bert_encoder_layer_11_attention_ln, l__self___bert_encoder_layer_11_ln, l__self___bert_encoder_layer_13_attention_ln, l__self___bert_encoder_layer_13_ln, l__self___bert_encoder_layer_15_attention_ln, l__self___bert_encoder_layer_15_ln, l__self___bert_encoder_layer_17_attention_ln, l__self___bert_encoder_layer_17_ln, l__self___bert_encoder_layer_19_attention_ln, l__self___bert_encoder_layer_19_ln, l__self___bert_encoder_layer_1_attention_ln, l__self___bert_encoder_layer_1_ln, l__self___bert_encoder_layer_21_attention_ln, l__self___bert_encoder_layer_21_ln, l__self___bert_encoder_layer_23_attention_ln, l__self___bert_encoder_layer_23_ln, l__self___bert_encoder_layer_3_attention_ln, l__self___bert_encoder_layer_3_ln, l__self___bert_encoder_layer_5_attention_ln, l__self___bert_encoder_layer_5_ln, l__self___bert_encoder_layer_7_attention_ln, l__self___bert_encoder_layer_7_ln, l__self___bert_encoder_layer_9_attention_ln, l__self___bert_encoder_layer_9_ln, l__self___bert_encoder_ln], Original ATen: [aten._to_copy, aten.add, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_11 => add_29
# add_12 => add_33
# add_14 => add_37
# add_15 => add_41
# add_17 => add_45
# add_18 => add_49
# add_2 => add_5
# add_20 => add_53
# add_21 => add_57
# add_23 => add_61
# add_24 => add_65
# add_26 => add_69
# add_27 => add_73
# add_29 => add_77
# add_3 => add_9
# add_30 => add_81
# add_32 => add_85
# add_33 => add_89
# add_35 => add_93
# add_36 => add_97
# add_38 => add_101
# add_39 => add_105
# add_41 => add_109
# add_42 => add_113
# add_44 => add_117
# add_45 => add_121
# add_47 => add_125
# add_48 => add_129
# add_5 => add_13
# add_50 => add_133
# add_51 => add_137
# add_53 => add_141
# add_54 => add_145
# add_56 => add_149
# add_57 => add_153
# add_59 => add_157
# add_6 => add_17
# add_60 => add_161
# add_62 => add_165
# add_63 => add_169
# add_65 => add_173
# add_66 => add_177
# add_68 => add_181
# add_69 => add_185
# add_71 => add_189
# add_72 => add_193
# add_8 => add_21
# add_9 => add_25
# l__self___bert_encoder_layer_11_attention_ln => mul_146, sub_34
# l__self___bert_encoder_layer_11_ln => mul_152, sub_36
# l__self___bert_encoder_layer_13_attention_ln => mul_172, sub_40
# l__self___bert_encoder_layer_13_ln => mul_178, sub_42
# l__self___bert_encoder_layer_15_attention_ln => mul_198, sub_46
# l__self___bert_encoder_layer_15_ln => mul_204, sub_48
# l__self___bert_encoder_layer_17_attention_ln => mul_224, sub_52
# l__self___bert_encoder_layer_17_ln => mul_230, sub_54
# l__self___bert_encoder_layer_19_attention_ln => mul_250, sub_58
# l__self___bert_encoder_layer_19_ln => mul_256, sub_60
# l__self___bert_encoder_layer_1_attention_ln => mul_16, sub_4
# l__self___bert_encoder_layer_1_ln => mul_22, sub_6
# l__self___bert_encoder_layer_21_attention_ln => mul_276, sub_64
# l__self___bert_encoder_layer_21_ln => mul_282, sub_66
# l__self___bert_encoder_layer_23_attention_ln => mul_302, sub_70
# l__self___bert_encoder_layer_23_ln => mul_308, sub_72
# l__self___bert_encoder_layer_3_attention_ln => mul_42, sub_10
# l__self___bert_encoder_layer_3_ln => mul_48, sub_12
# l__self___bert_encoder_layer_5_attention_ln => mul_68, sub_16
# l__self___bert_encoder_layer_5_ln => mul_74, sub_18
# l__self___bert_encoder_layer_7_attention_ln => mul_94, sub_22
# l__self___bert_encoder_layer_7_ln => mul_100, sub_24
# l__self___bert_encoder_layer_9_attention_ln => mul_120, sub_28
# l__self___bert_encoder_layer_9_ln => mul_126, sub_30
# l__self___bert_encoder_ln => mul_315, sub_73
triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp32', 12: '*fp32', 13: '*fp16', 14: '*fp32', 15: '*fp32', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp32', 20: '*fp32', 21: '*fp16', 22: '*fp32', 23: '*fp32', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp32', 28: '*fp32', 29: '*fp16', 30: '*fp32', 31: '*fp32', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp32', 36: '*fp32', 37: '*fp16', 38: '*fp32', 39: '*fp32', 40: '*fp16', 41: '*fp16', 42: '*fp16', 43: '*fp32', 44: '*fp32', 45: '*fp16', 46: '*fp32', 47: '*fp32', 48: '*fp16', 49: '*fp16', 50: '*fp16', 51: '*fp32', 52: '*fp32', 53: '*fp16', 54: '*fp32', 55: '*fp32', 56: '*fp16', 57: '*fp16', 58: '*fp16', 59: '*fp32', 60: '*fp32', 61: '*fp16', 62: '*fp32', 63: '*fp32', 64: '*fp16', 65: '*fp16', 66: '*fp16', 67: '*fp32', 68: '*fp32', 69: '*fp16', 70: '*fp32', 71: '*fp32', 72: '*fp16', 73: '*fp16', 74: '*fp16', 75: '*fp32', 76: '*fp32', 77: '*fp16', 78: '*fp32', 79: '*fp32', 80: '*fp16', 81: '*fp16', 82: '*fp16', 83: '*fp32', 84: '*fp32', 85: '*fp16', 86: '*fp32', 87: '*fp32', 88: '*fp16', 89: '*fp16', 90: '*fp16', 91: '*fp32', 92: '*fp32', 93: '*fp16', 94: '*fp32', 95: '*fp32', 96: '*fp16', 97: '*fp32', 98: '*fp32', 99: '*fp16', 100: '*fp32', 101: '*i1', 102: '*fp32', 103: '*fp32', 104: '*fp32', 105: '*fp32', 106: '*fp32', 107: '*fp32', 108: '*fp32', 109: '*fp32', 110: '*fp32', 111: '*fp32', 112: '*fp32', 113: '*fp32', 114: '*fp32', 115: '*fp32', 116: '*fp32', 117: '*fp32', 118: '*fp32', 119: '*fp32', 120: '*fp32', 121: '*fp32', 122: '*fp32', 123: '*fp32', 124: '*fp32', 125: '*fp32', 126: '*fp32', 127: '*fp32', 128: '*fp32', 129: '*fp32', 130: '*fp32', 131: '*fp32', 132: '*fp32', 133: '*fp32', 134: '*fp32', 135: '*fp32', 136: '*fp32', 137: '*fp32', 138: '*fp32', 139: '*fp16', 140: 'i32', 141: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81, in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88, in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95, in_ptr96, in_ptr97, in_ptr98, in_ptr99, in_ptr100, in_ptr101, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr38, out_ptr39, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr9 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr10 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr13 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr15 + (x0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr16 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp41 = tl.load(in_ptr17 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp44 = tl.load(in_ptr18 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp47 = tl.load(in_ptr19 + (x0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr20 + (x0), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr21 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp54 = tl.load(in_ptr22 + (x0), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr23 + (x0), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr24 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp61 = tl.load(in_ptr25 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp64 = tl.load(in_ptr26 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp67 = tl.load(in_ptr27 + (x0), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr28 + (x0), None, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr29 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp74 = tl.load(in_ptr30 + (x0), None, eviction_policy='evict_last')
    tmp76 = tl.load(in_ptr31 + (x0), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr32 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp81 = tl.load(in_ptr33 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp84 = tl.load(in_ptr34 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp87 = tl.load(in_ptr35 + (x0), None, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr36 + (x0), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr37 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp94 = tl.load(in_ptr38 + (x0), None, eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr39 + (x0), None, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr40 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp101 = tl.load(in_ptr41 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp104 = tl.load(in_ptr42 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp107 = tl.load(in_ptr43 + (x0), None, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr44 + (x0), None, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr45 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp114 = tl.load(in_ptr46 + (x0), None, eviction_policy='evict_last')
    tmp116 = tl.load(in_ptr47 + (x0), None, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr48 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp121 = tl.load(in_ptr49 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp124 = tl.load(in_ptr50 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp127 = tl.load(in_ptr51 + (x0), None, eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr52 + (x0), None, eviction_policy='evict_last')
    tmp131 = tl.load(in_ptr53 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp134 = tl.load(in_ptr54 + (x0), None, eviction_policy='evict_last')
    tmp136 = tl.load(in_ptr55 + (x0), None, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr56 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp141 = tl.load(in_ptr57 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp144 = tl.load(in_ptr58 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp147 = tl.load(in_ptr59 + (x0), None, eviction_policy='evict_last')
    tmp149 = tl.load(in_ptr60 + (x0), None, eviction_policy='evict_last')
    tmp151 = tl.load(in_ptr61 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp154 = tl.load(in_ptr62 + (x0), None, eviction_policy='evict_last')
    tmp156 = tl.load(in_ptr63 + (x0), None, eviction_policy='evict_last')
    tmp158 = tl.load(in_ptr64 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp161 = tl.load(in_ptr65 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp164 = tl.load(in_ptr66 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp167 = tl.load(in_ptr67 + (x0), None, eviction_policy='evict_last')
    tmp169 = tl.load(in_ptr68 + (x0), None, eviction_policy='evict_last')
    tmp171 = tl.load(in_ptr69 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp174 = tl.load(in_ptr70 + (x0), None, eviction_policy='evict_last')
    tmp176 = tl.load(in_ptr71 + (x0), None, eviction_policy='evict_last')
    tmp178 = tl.load(in_ptr72 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp181 = tl.load(in_ptr73 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp184 = tl.load(in_ptr74 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp187 = tl.load(in_ptr75 + (x0), None, eviction_policy='evict_last')
    tmp189 = tl.load(in_ptr76 + (x0), None, eviction_policy='evict_last')
    tmp191 = tl.load(in_ptr77 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp194 = tl.load(in_ptr78 + (x0), None, eviction_policy='evict_last')
    tmp196 = tl.load(in_ptr79 + (x0), None, eviction_policy='evict_last')
    tmp198 = tl.load(in_ptr80 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp201 = tl.load(in_ptr81 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp204 = tl.load(in_ptr82 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp207 = tl.load(in_ptr83 + (x0), None, eviction_policy='evict_last')
    tmp209 = tl.load(in_ptr84 + (x0), None, eviction_policy='evict_last')
    tmp211 = tl.load(in_ptr85 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp214 = tl.load(in_ptr86 + (x0), None, eviction_policy='evict_last')
    tmp216 = tl.load(in_ptr87 + (x0), None, eviction_policy='evict_last')
    tmp218 = tl.load(in_ptr88 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp221 = tl.load(in_ptr89 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp224 = tl.load(in_ptr90 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp227 = tl.load(in_ptr91 + (x0), None, eviction_policy='evict_last')
    tmp229 = tl.load(in_ptr92 + (x0), None, eviction_policy='evict_last')
    tmp231 = tl.load(in_ptr93 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp234 = tl.load(in_ptr94 + (x0), None, eviction_policy='evict_last')
    tmp236 = tl.load(in_ptr95 + (x0), None, eviction_policy='evict_last')
    tmp238 = tl.load(in_ptr96 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp241 = tl.load(in_ptr97 + (x0), None, eviction_policy='evict_last')
    tmp243 = tl.load(in_ptr98 + (x0), None, eviction_policy='evict_last')
    tmp245 = tl.load(in_ptr99 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp247 = tl.load(in_ptr100 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp266 = tl.load(in_ptr101 + (r1 + (1024*x0)), rmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp6 + tmp12
    tmp15 = tmp13 - tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp13 + tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 + tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 + tmp25
    tmp28 = tmp26 - tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp26 + tmp32
    tmp35 = tmp33 - tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp33 + tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 + tmp42
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp43 + tmp45
    tmp48 = tmp46 - tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp46 + tmp52
    tmp55 = tmp53 - tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp53 + tmp59
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp60 + tmp62
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp63 + tmp65
    tmp68 = tmp66 - tmp67
    tmp70 = tmp68 * tmp69
    tmp72 = tmp71.to(tl.float32)
    tmp73 = tmp66 + tmp72
    tmp75 = tmp73 - tmp74
    tmp77 = tmp75 * tmp76
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tmp73 + tmp79
    tmp82 = tmp81.to(tl.float32)
    tmp83 = tmp80 + tmp82
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tmp83 + tmp85
    tmp88 = tmp86 - tmp87
    tmp90 = tmp88 * tmp89
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tmp86 + tmp92
    tmp95 = tmp93 - tmp94
    tmp97 = tmp95 * tmp96
    tmp99 = tmp98.to(tl.float32)
    tmp100 = tmp93 + tmp99
    tmp102 = tmp101.to(tl.float32)
    tmp103 = tmp100 + tmp102
    tmp105 = tmp104.to(tl.float32)
    tmp106 = tmp103 + tmp105
    tmp108 = tmp106 - tmp107
    tmp110 = tmp108 * tmp109
    tmp112 = tmp111.to(tl.float32)
    tmp113 = tmp106 + tmp112
    tmp115 = tmp113 - tmp114
    tmp117 = tmp115 * tmp116
    tmp119 = tmp118.to(tl.float32)
    tmp120 = tmp113 + tmp119
    tmp122 = tmp121.to(tl.float32)
    tmp123 = tmp120 + tmp122
    tmp125 = tmp124.to(tl.float32)
    tmp126 = tmp123 + tmp125
    tmp128 = tmp126 - tmp127
    tmp130 = tmp128 * tmp129
    tmp132 = tmp131.to(tl.float32)
    tmp133 = tmp126 + tmp132
    tmp135 = tmp133 - tmp134
    tmp137 = tmp135 * tmp136
    tmp139 = tmp138.to(tl.float32)
    tmp140 = tmp133 + tmp139
    tmp142 = tmp141.to(tl.float32)
    tmp143 = tmp140 + tmp142
    tmp145 = tmp144.to(tl.float32)
    tmp146 = tmp143 + tmp145
    tmp148 = tmp146 - tmp147
    tmp150 = tmp148 * tmp149
    tmp152 = tmp151.to(tl.float32)
    tmp153 = tmp146 + tmp152
    tmp155 = tmp153 - tmp154
    tmp157 = tmp155 * tmp156
    tmp159 = tmp158.to(tl.float32)
    tmp160 = tmp153 + tmp159
    tmp162 = tmp161.to(tl.float32)
    tmp163 = tmp160 + tmp162
    tmp165 = tmp164.to(tl.float32)
    tmp166 = tmp163 + tmp165
    tmp168 = tmp166 - tmp167
    tmp170 = tmp168 * tmp169
    tmp172 = tmp171.to(tl.float32)
    tmp173 = tmp166 + tmp172
    tmp175 = tmp173 - tmp174
    tmp177 = tmp175 * tmp176
    tmp179 = tmp178.to(tl.float32)
    tmp180 = tmp173 + tmp179
    tmp182 = tmp181.to(tl.float32)
    tmp183 = tmp180 + tmp182
    tmp185 = tmp184.to(tl.float32)
    tmp186 = tmp183 + tmp185
    tmp188 = tmp186 - tmp187
    tmp190 = tmp188 * tmp189
    tmp192 = tmp191.to(tl.float32)
    tmp193 = tmp186 + tmp192
    tmp195 = tmp193 - tmp194
    tmp197 = tmp195 * tmp196
    tmp199 = tmp198.to(tl.float32)
    tmp200 = tmp193 + tmp199
    tmp202 = tmp201.to(tl.float32)
    tmp203 = tmp200 + tmp202
    tmp205 = tmp204.to(tl.float32)
    tmp206 = tmp203 + tmp205
    tmp208 = tmp206 - tmp207
    tmp210 = tmp208 * tmp209
    tmp212 = tmp211.to(tl.float32)
    tmp213 = tmp206 + tmp212
    tmp215 = tmp213 - tmp214
    tmp217 = tmp215 * tmp216
    tmp219 = tmp218.to(tl.float32)
    tmp220 = tmp213 + tmp219
    tmp222 = tmp221.to(tl.float32)
    tmp223 = tmp220 + tmp222
    tmp225 = tmp224.to(tl.float32)
    tmp226 = tmp223 + tmp225
    tmp228 = tmp226 - tmp227
    tmp230 = tmp228 * tmp229
    tmp232 = tmp231.to(tl.float32)
    tmp233 = tmp226 + tmp232
    tmp235 = tmp233 - tmp234
    tmp237 = tmp235 * tmp236
    tmp239 = tmp238.to(tl.float32)
    tmp240 = tmp233 + tmp239
    tmp242 = tmp240 - tmp241
    tmp244 = tmp242 * tmp243
    tmp246 = tmp245.to(tl.float32)
    tmp248 = tmp246 * tmp247
    tmp249 = tl.broadcast_to(tmp248, [RBLOCK])
    tmp251 = tl.where(rmask, tmp249, 0)
    tmp252 = triton_helpers.promote_to_tensor(tl.sum(tmp251, 0))
    tmp253 = tmp248 * tmp244
    tmp254 = tl.broadcast_to(tmp253, [RBLOCK])
    tmp256 = tl.where(rmask, tmp254, 0)
    tmp257 = triton_helpers.promote_to_tensor(tl.sum(tmp256, 0))
    tmp258 = 1024.0
    tmp259 = tmp243 / tmp258
    tmp260 = tmp248 * tmp258
    tmp261 = tmp260 - tmp252
    tmp262 = tmp244 * tmp257
    tmp263 = tmp261 - tmp262
    tmp264 = tmp259 * tmp263
    tmp265 = tmp264.to(tl.float32)
    tmp267 = tmp266.to(tl.float32)
    tmp268 = 1.1111111111111112
    tmp269 = tmp267 * tmp268
    tmp270 = tmp265 * tmp269
    tl.store(out_ptr0 + (r1 + (1024*x0)), tmp10, rmask)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp17, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp20, rmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp30, rmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp37, rmask)
    tl.store(out_ptr5 + (r1 + (1024*x0)), tmp40, rmask)
    tl.store(out_ptr6 + (r1 + (1024*x0)), tmp50, rmask)
    tl.store(out_ptr7 + (r1 + (1024*x0)), tmp57, rmask)
    tl.store(out_ptr8 + (r1 + (1024*x0)), tmp60, rmask)
    tl.store(out_ptr9 + (r1 + (1024*x0)), tmp70, rmask)
    tl.store(out_ptr10 + (r1 + (1024*x0)), tmp77, rmask)
    tl.store(out_ptr11 + (r1 + (1024*x0)), tmp80, rmask)
    tl.store(out_ptr12 + (r1 + (1024*x0)), tmp90, rmask)
    tl.store(out_ptr13 + (r1 + (1024*x0)), tmp97, rmask)
    tl.store(out_ptr14 + (r1 + (1024*x0)), tmp100, rmask)
    tl.store(out_ptr15 + (r1 + (1024*x0)), tmp110, rmask)
    tl.store(out_ptr16 + (r1 + (1024*x0)), tmp117, rmask)
    tl.store(out_ptr17 + (r1 + (1024*x0)), tmp120, rmask)
    tl.store(out_ptr18 + (r1 + (1024*x0)), tmp130, rmask)
    tl.store(out_ptr19 + (r1 + (1024*x0)), tmp137, rmask)
    tl.store(out_ptr20 + (r1 + (1024*x0)), tmp140, rmask)
    tl.store(out_ptr21 + (r1 + (1024*x0)), tmp150, rmask)
    tl.store(out_ptr22 + (r1 + (1024*x0)), tmp157, rmask)
    tl.store(out_ptr23 + (r1 + (1024*x0)), tmp160, rmask)
    tl.store(out_ptr24 + (r1 + (1024*x0)), tmp170, rmask)
    tl.store(out_ptr25 + (r1 + (1024*x0)), tmp177, rmask)
    tl.store(out_ptr26 + (r1 + (1024*x0)), tmp180, rmask)
    tl.store(out_ptr27 + (r1 + (1024*x0)), tmp190, rmask)
    tl.store(out_ptr28 + (r1 + (1024*x0)), tmp197, rmask)
    tl.store(out_ptr29 + (r1 + (1024*x0)), tmp200, rmask)
    tl.store(out_ptr30 + (r1 + (1024*x0)), tmp210, rmask)
    tl.store(out_ptr31 + (r1 + (1024*x0)), tmp217, rmask)
    tl.store(out_ptr32 + (r1 + (1024*x0)), tmp220, rmask)
    tl.store(out_ptr33 + (r1 + (1024*x0)), tmp230, rmask)
    tl.store(out_ptr34 + (r1 + (1024*x0)), tmp237, rmask)
    tl.store(out_ptr35 + (r1 + (1024*x0)), tmp244, rmask)
    tl.store(out_ptr38 + (r1 + (1024*x0)), tmp264, rmask)
    tl.store(out_ptr39 + (r1 + (1024*x0)), tmp270, rmask)
''')
