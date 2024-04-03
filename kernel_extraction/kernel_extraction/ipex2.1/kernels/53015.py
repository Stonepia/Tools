

# Original file: ./AlbertForQuestionAnswering__0_backward_135.1/AlbertForQuestionAnswering__0_backward_135.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/bf/cbfda4z7z7o77ef5qwmrph2uofiiqp4qpagcjcqefrhyfoysv527.py
# Source Nodes: [l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_1, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_10, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_11, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_2, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_3, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_4, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_5, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_6, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_7, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_8, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.sum]
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm => convert_element_type_5
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_1 => convert_element_type_11
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_10 => convert_element_type_65
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_11 => convert_element_type_71
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_2 => convert_element_type_17
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_3 => convert_element_type_23
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_4 => convert_element_type_29
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_5 => convert_element_type_35
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_6 => convert_element_type_41
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_7 => convert_element_type_47
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_8 => convert_element_type_53
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_9 => convert_element_type_59
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_sum_14 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_sum_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*fp32', 12: '*fp32', 13: '*bf16', 14: '*bf16', 15: '*bf16', 16: '*fp32', 17: '*fp32', 18: '*bf16', 19: '*bf16', 20: '*bf16', 21: '*fp32', 22: '*fp32', 23: '*bf16', 24: '*bf16', 25: '*bf16', 26: '*fp32', 27: '*fp32', 28: '*bf16', 29: '*bf16', 30: '*bf16', 31: '*fp32', 32: '*fp32', 33: '*bf16', 34: '*bf16', 35: '*bf16', 36: '*fp32', 37: '*fp32', 38: '*bf16', 39: '*bf16', 40: '*bf16', 41: '*fp32', 42: '*fp32', 43: '*bf16', 44: '*bf16', 45: '*bf16', 46: '*fp32', 47: '*fp32', 48: '*bf16', 49: '*bf16', 50: '*bf16', 51: '*fp32', 52: '*fp32', 53: '*bf16', 54: '*bf16', 55: '*bf16', 56: '*fp32', 57: '*fp32', 58: '*bf16', 59: '*bf16', 60: '*bf16', 61: '*fp32', 62: '*fp32', 63: 'i32', 64: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_sum_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_sum_14(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp35 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp42 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp55 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp58 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp62 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp75 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp78 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp82 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp95 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp98 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp102 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp115 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp118 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp122 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp135 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp138 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp142 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp155 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp158 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp162 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp175 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp178 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp182 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp195 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp198 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp202 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp215 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp218 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp222 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp235 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp238 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr8 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr9 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp40 = tl.load(in_ptr10 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp44 = tl.load(in_ptr11 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp47 = tl.load(in_ptr12 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp49 = tl.load(in_ptr13 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.load(in_ptr14 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp60 = tl.load(in_ptr15 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp64 = tl.load(in_ptr16 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp67 = tl.load(in_ptr17 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp69 = tl.load(in_ptr18 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp71 = tl.load(in_ptr19 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp80 = tl.load(in_ptr20 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp84 = tl.load(in_ptr21 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp87 = tl.load(in_ptr22 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp89 = tl.load(in_ptr23 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp91 = tl.load(in_ptr24 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp100 = tl.load(in_ptr25 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp104 = tl.load(in_ptr26 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp107 = tl.load(in_ptr27 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp109 = tl.load(in_ptr28 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp111 = tl.load(in_ptr29 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp120 = tl.load(in_ptr30 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp124 = tl.load(in_ptr31 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp127 = tl.load(in_ptr32 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp129 = tl.load(in_ptr33 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp131 = tl.load(in_ptr34 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp140 = tl.load(in_ptr35 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp144 = tl.load(in_ptr36 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp147 = tl.load(in_ptr37 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp149 = tl.load(in_ptr38 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp151 = tl.load(in_ptr39 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp160 = tl.load(in_ptr40 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp164 = tl.load(in_ptr41 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp167 = tl.load(in_ptr42 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp169 = tl.load(in_ptr43 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp171 = tl.load(in_ptr44 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp180 = tl.load(in_ptr45 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp184 = tl.load(in_ptr46 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp187 = tl.load(in_ptr47 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp189 = tl.load(in_ptr48 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp191 = tl.load(in_ptr49 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp200 = tl.load(in_ptr50 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp204 = tl.load(in_ptr51 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp207 = tl.load(in_ptr52 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp209 = tl.load(in_ptr53 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp211 = tl.load(in_ptr54 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp220 = tl.load(in_ptr55 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp224 = tl.load(in_ptr56 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp227 = tl.load(in_ptr57 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp229 = tl.load(in_ptr58 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp231 = tl.load(in_ptr59 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp10 = tmp8 - tmp9
        tmp12 = tmp10 * tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
        tmp17 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
        tmp25 = tmp20 + tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp30 = tmp28 - tmp29
        tmp32 = tmp30 * tmp31
        tmp33 = tmp26 * tmp32
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(rmask, tmp36, _tmp35)
        tmp37 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask, tmp39, _tmp38)
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, RBLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(rmask, tmp43, _tmp42)
        tmp45 = tmp40 + tmp44
        tmp46 = tmp45.to(tl.float32)
        tmp48 = tmp47.to(tl.float32)
        tmp50 = tmp48 - tmp49
        tmp52 = tmp50 * tmp51
        tmp53 = tmp46 * tmp52
        tmp54 = tl.broadcast_to(tmp53, [XBLOCK, RBLOCK])
        tmp56 = _tmp55 + tmp54
        _tmp55 = tl.where(rmask, tmp56, _tmp55)
        tmp57 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
        tmp59 = _tmp58 + tmp57
        _tmp58 = tl.where(rmask, tmp59, _tmp58)
        tmp61 = tl.broadcast_to(tmp60, [XBLOCK, RBLOCK])
        tmp63 = _tmp62 + tmp61
        _tmp62 = tl.where(rmask, tmp63, _tmp62)
        tmp65 = tmp60 + tmp64
        tmp66 = tmp65.to(tl.float32)
        tmp68 = tmp67.to(tl.float32)
        tmp70 = tmp68 - tmp69
        tmp72 = tmp70 * tmp71
        tmp73 = tmp66 * tmp72
        tmp74 = tl.broadcast_to(tmp73, [XBLOCK, RBLOCK])
        tmp76 = _tmp75 + tmp74
        _tmp75 = tl.where(rmask, tmp76, _tmp75)
        tmp77 = tl.broadcast_to(tmp66, [XBLOCK, RBLOCK])
        tmp79 = _tmp78 + tmp77
        _tmp78 = tl.where(rmask, tmp79, _tmp78)
        tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
        tmp83 = _tmp82 + tmp81
        _tmp82 = tl.where(rmask, tmp83, _tmp82)
        tmp85 = tmp80 + tmp84
        tmp86 = tmp85.to(tl.float32)
        tmp88 = tmp87.to(tl.float32)
        tmp90 = tmp88 - tmp89
        tmp92 = tmp90 * tmp91
        tmp93 = tmp86 * tmp92
        tmp94 = tl.broadcast_to(tmp93, [XBLOCK, RBLOCK])
        tmp96 = _tmp95 + tmp94
        _tmp95 = tl.where(rmask, tmp96, _tmp95)
        tmp97 = tl.broadcast_to(tmp86, [XBLOCK, RBLOCK])
        tmp99 = _tmp98 + tmp97
        _tmp98 = tl.where(rmask, tmp99, _tmp98)
        tmp101 = tl.broadcast_to(tmp100, [XBLOCK, RBLOCK])
        tmp103 = _tmp102 + tmp101
        _tmp102 = tl.where(rmask, tmp103, _tmp102)
        tmp105 = tmp100 + tmp104
        tmp106 = tmp105.to(tl.float32)
        tmp108 = tmp107.to(tl.float32)
        tmp110 = tmp108 - tmp109
        tmp112 = tmp110 * tmp111
        tmp113 = tmp106 * tmp112
        tmp114 = tl.broadcast_to(tmp113, [XBLOCK, RBLOCK])
        tmp116 = _tmp115 + tmp114
        _tmp115 = tl.where(rmask, tmp116, _tmp115)
        tmp117 = tl.broadcast_to(tmp106, [XBLOCK, RBLOCK])
        tmp119 = _tmp118 + tmp117
        _tmp118 = tl.where(rmask, tmp119, _tmp118)
        tmp121 = tl.broadcast_to(tmp120, [XBLOCK, RBLOCK])
        tmp123 = _tmp122 + tmp121
        _tmp122 = tl.where(rmask, tmp123, _tmp122)
        tmp125 = tmp120 + tmp124
        tmp126 = tmp125.to(tl.float32)
        tmp128 = tmp127.to(tl.float32)
        tmp130 = tmp128 - tmp129
        tmp132 = tmp130 * tmp131
        tmp133 = tmp126 * tmp132
        tmp134 = tl.broadcast_to(tmp133, [XBLOCK, RBLOCK])
        tmp136 = _tmp135 + tmp134
        _tmp135 = tl.where(rmask, tmp136, _tmp135)
        tmp137 = tl.broadcast_to(tmp126, [XBLOCK, RBLOCK])
        tmp139 = _tmp138 + tmp137
        _tmp138 = tl.where(rmask, tmp139, _tmp138)
        tmp141 = tl.broadcast_to(tmp140, [XBLOCK, RBLOCK])
        tmp143 = _tmp142 + tmp141
        _tmp142 = tl.where(rmask, tmp143, _tmp142)
        tmp145 = tmp140 + tmp144
        tmp146 = tmp145.to(tl.float32)
        tmp148 = tmp147.to(tl.float32)
        tmp150 = tmp148 - tmp149
        tmp152 = tmp150 * tmp151
        tmp153 = tmp146 * tmp152
        tmp154 = tl.broadcast_to(tmp153, [XBLOCK, RBLOCK])
        tmp156 = _tmp155 + tmp154
        _tmp155 = tl.where(rmask, tmp156, _tmp155)
        tmp157 = tl.broadcast_to(tmp146, [XBLOCK, RBLOCK])
        tmp159 = _tmp158 + tmp157
        _tmp158 = tl.where(rmask, tmp159, _tmp158)
        tmp161 = tl.broadcast_to(tmp160, [XBLOCK, RBLOCK])
        tmp163 = _tmp162 + tmp161
        _tmp162 = tl.where(rmask, tmp163, _tmp162)
        tmp165 = tmp160 + tmp164
        tmp166 = tmp165.to(tl.float32)
        tmp168 = tmp167.to(tl.float32)
        tmp170 = tmp168 - tmp169
        tmp172 = tmp170 * tmp171
        tmp173 = tmp166 * tmp172
        tmp174 = tl.broadcast_to(tmp173, [XBLOCK, RBLOCK])
        tmp176 = _tmp175 + tmp174
        _tmp175 = tl.where(rmask, tmp176, _tmp175)
        tmp177 = tl.broadcast_to(tmp166, [XBLOCK, RBLOCK])
        tmp179 = _tmp178 + tmp177
        _tmp178 = tl.where(rmask, tmp179, _tmp178)
        tmp181 = tl.broadcast_to(tmp180, [XBLOCK, RBLOCK])
        tmp183 = _tmp182 + tmp181
        _tmp182 = tl.where(rmask, tmp183, _tmp182)
        tmp185 = tmp180 + tmp184
        tmp186 = tmp185.to(tl.float32)
        tmp188 = tmp187.to(tl.float32)
        tmp190 = tmp188 - tmp189
        tmp192 = tmp190 * tmp191
        tmp193 = tmp186 * tmp192
        tmp194 = tl.broadcast_to(tmp193, [XBLOCK, RBLOCK])
        tmp196 = _tmp195 + tmp194
        _tmp195 = tl.where(rmask, tmp196, _tmp195)
        tmp197 = tl.broadcast_to(tmp186, [XBLOCK, RBLOCK])
        tmp199 = _tmp198 + tmp197
        _tmp198 = tl.where(rmask, tmp199, _tmp198)
        tmp201 = tl.broadcast_to(tmp200, [XBLOCK, RBLOCK])
        tmp203 = _tmp202 + tmp201
        _tmp202 = tl.where(rmask, tmp203, _tmp202)
        tmp205 = tmp200 + tmp204
        tmp206 = tmp205.to(tl.float32)
        tmp208 = tmp207.to(tl.float32)
        tmp210 = tmp208 - tmp209
        tmp212 = tmp210 * tmp211
        tmp213 = tmp206 * tmp212
        tmp214 = tl.broadcast_to(tmp213, [XBLOCK, RBLOCK])
        tmp216 = _tmp215 + tmp214
        _tmp215 = tl.where(rmask, tmp216, _tmp215)
        tmp217 = tl.broadcast_to(tmp206, [XBLOCK, RBLOCK])
        tmp219 = _tmp218 + tmp217
        _tmp218 = tl.where(rmask, tmp219, _tmp218)
        tmp221 = tl.broadcast_to(tmp220, [XBLOCK, RBLOCK])
        tmp223 = _tmp222 + tmp221
        _tmp222 = tl.where(rmask, tmp223, _tmp222)
        tmp225 = tmp220 + tmp224
        tmp226 = tmp225.to(tl.float32)
        tmp228 = tmp227.to(tl.float32)
        tmp230 = tmp228 - tmp229
        tmp232 = tmp230 * tmp231
        tmp233 = tmp226 * tmp232
        tmp234 = tl.broadcast_to(tmp233, [XBLOCK, RBLOCK])
        tmp236 = _tmp235 + tmp234
        _tmp235 = tl.where(rmask, tmp236, _tmp235)
        tmp237 = tl.broadcast_to(tmp226, [XBLOCK, RBLOCK])
        tmp239 = _tmp238 + tmp237
        _tmp238 = tl.where(rmask, tmp239, _tmp238)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp55 = tl.sum(_tmp55, 1)[:, None]
    tmp58 = tl.sum(_tmp58, 1)[:, None]
    tmp62 = tl.sum(_tmp62, 1)[:, None]
    tmp75 = tl.sum(_tmp75, 1)[:, None]
    tmp78 = tl.sum(_tmp78, 1)[:, None]
    tmp82 = tl.sum(_tmp82, 1)[:, None]
    tmp95 = tl.sum(_tmp95, 1)[:, None]
    tmp98 = tl.sum(_tmp98, 1)[:, None]
    tmp102 = tl.sum(_tmp102, 1)[:, None]
    tmp115 = tl.sum(_tmp115, 1)[:, None]
    tmp118 = tl.sum(_tmp118, 1)[:, None]
    tmp122 = tl.sum(_tmp122, 1)[:, None]
    tmp135 = tl.sum(_tmp135, 1)[:, None]
    tmp138 = tl.sum(_tmp138, 1)[:, None]
    tmp142 = tl.sum(_tmp142, 1)[:, None]
    tmp155 = tl.sum(_tmp155, 1)[:, None]
    tmp158 = tl.sum(_tmp158, 1)[:, None]
    tmp162 = tl.sum(_tmp162, 1)[:, None]
    tmp175 = tl.sum(_tmp175, 1)[:, None]
    tmp178 = tl.sum(_tmp178, 1)[:, None]
    tmp182 = tl.sum(_tmp182, 1)[:, None]
    tmp195 = tl.sum(_tmp195, 1)[:, None]
    tmp198 = tl.sum(_tmp198, 1)[:, None]
    tmp202 = tl.sum(_tmp202, 1)[:, None]
    tmp215 = tl.sum(_tmp215, 1)[:, None]
    tmp218 = tl.sum(_tmp218, 1)[:, None]
    tmp222 = tl.sum(_tmp222, 1)[:, None]
    tmp235 = tl.sum(_tmp235, 1)[:, None]
    tmp238 = tl.sum(_tmp238, 1)[:, None]
    tmp240 = tmp15.to(tl.float32)
    tmp241 = tmp35.to(tl.float32)
    tmp242 = tmp240 + tmp241
    tmp243 = tmp55.to(tl.float32)
    tmp244 = tmp242 + tmp243
    tmp245 = tmp75.to(tl.float32)
    tmp246 = tmp244 + tmp245
    tmp247 = tmp95.to(tl.float32)
    tmp248 = tmp246 + tmp247
    tmp249 = tmp115.to(tl.float32)
    tmp250 = tmp248 + tmp249
    tmp251 = tmp155.to(tl.float32)
    tmp252 = tmp250 + tmp251
    tmp253 = tmp195.to(tl.float32)
    tmp254 = tmp252 + tmp253
    tmp255 = tmp235.to(tl.float32)
    tmp256 = tmp254 + tmp255
    tmp257 = tmp135.to(tl.float32)
    tmp258 = tmp256 + tmp257
    tmp259 = tmp175.to(tl.float32)
    tmp260 = tmp258 + tmp259
    tmp261 = tmp215.to(tl.float32)
    tmp262 = tmp260 + tmp261
    tmp263 = tmp18.to(tl.float32)
    tmp264 = tmp38.to(tl.float32)
    tmp265 = tmp263 + tmp264
    tmp266 = tmp58.to(tl.float32)
    tmp267 = tmp265 + tmp266
    tmp268 = tmp78.to(tl.float32)
    tmp269 = tmp267 + tmp268
    tmp270 = tmp98.to(tl.float32)
    tmp271 = tmp269 + tmp270
    tmp272 = tmp118.to(tl.float32)
    tmp273 = tmp271 + tmp272
    tmp274 = tmp158.to(tl.float32)
    tmp275 = tmp273 + tmp274
    tmp276 = tmp198.to(tl.float32)
    tmp277 = tmp275 + tmp276
    tmp278 = tmp238.to(tl.float32)
    tmp279 = tmp277 + tmp278
    tmp280 = tmp138.to(tl.float32)
    tmp281 = tmp279 + tmp280
    tmp282 = tmp178.to(tl.float32)
    tmp283 = tmp281 + tmp282
    tmp284 = tmp218.to(tl.float32)
    tmp285 = tmp283 + tmp284
    tmp286 = tmp2 + tmp22
    tmp287 = tmp286 + tmp42
    tmp288 = tmp287 + tmp62
    tmp289 = tmp288 + tmp82
    tmp290 = tmp289 + tmp102
    tmp291 = tmp290 + tmp142
    tmp292 = tmp291 + tmp182
    tmp293 = tmp292 + tmp222
    tmp294 = tmp293 + tmp122
    tmp295 = tmp294 + tmp162
    tmp296 = tmp295 + tmp202
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp262, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp285, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp296, None)
''')
