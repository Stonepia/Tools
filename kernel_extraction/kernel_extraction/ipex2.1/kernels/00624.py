

# Original file: ./AlbertForMaskedLM__0_backward_207.1/AlbertForMaskedLM__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/je/cjemitsso4r5s6wlfohql5khbl65hb2culzo5rr3rbh76eltthjh.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]

triton_red_fused_add_native_layer_norm_backward_sum_17 = async_compile.triton('triton_red_fused_add_native_layer_norm_backward_sum_17', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: 'i32', 62: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_sum_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_backward_sum_17(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp40 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp43 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp47 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp58 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp61 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp65 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp76 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp79 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp83 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp94 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp97 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp101 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp112 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp115 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp119 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp130 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp133 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp137 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp148 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp151 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp155 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp166 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp169 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp173 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp184 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp187 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp191 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp195 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp206 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp209 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr3 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr4 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr8 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr9 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr10 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr11 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp45 = tl.load(in_ptr12 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp49 = tl.load(in_ptr13 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp51 = tl.load(in_ptr14 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp53 = tl.load(in_ptr15 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp55 = tl.load(in_ptr16 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = tl.load(in_ptr17 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp67 = tl.load(in_ptr18 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp69 = tl.load(in_ptr19 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp71 = tl.load(in_ptr20 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp73 = tl.load(in_ptr21 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp81 = tl.load(in_ptr22 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp85 = tl.load(in_ptr23 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp87 = tl.load(in_ptr24 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp89 = tl.load(in_ptr25 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp91 = tl.load(in_ptr26 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp99 = tl.load(in_ptr27 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp103 = tl.load(in_ptr28 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp105 = tl.load(in_ptr29 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp107 = tl.load(in_ptr30 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp109 = tl.load(in_ptr31 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp117 = tl.load(in_ptr32 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp121 = tl.load(in_ptr33 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp123 = tl.load(in_ptr34 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp125 = tl.load(in_ptr35 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp127 = tl.load(in_ptr36 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp135 = tl.load(in_ptr37 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp139 = tl.load(in_ptr38 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp141 = tl.load(in_ptr39 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp143 = tl.load(in_ptr40 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp145 = tl.load(in_ptr41 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp153 = tl.load(in_ptr42 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp157 = tl.load(in_ptr43 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp159 = tl.load(in_ptr44 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp161 = tl.load(in_ptr45 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp163 = tl.load(in_ptr46 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp171 = tl.load(in_ptr47 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp175 = tl.load(in_ptr48 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp177 = tl.load(in_ptr49 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp179 = tl.load(in_ptr50 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp181 = tl.load(in_ptr51 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp189 = tl.load(in_ptr52 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp193 = tl.load(in_ptr53 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp197 = tl.load(in_ptr54 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp199 = tl.load(in_ptr55 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp201 = tl.load(in_ptr56 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp203 = tl.load(in_ptr57 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
        tmp14 = tmp9 + tmp13
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp20 = tmp18 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
        tmp24 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask, tmp26, _tmp25)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask, tmp30, _tmp29)
        tmp32 = tmp27 + tmp31
        tmp34 = tmp32 + tmp33
        tmp36 = tmp34 + tmp35
        tmp38 = tmp36 * tmp37
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
        tmp41 = _tmp40 + tmp39
        _tmp40 = tl.where(rmask, tmp41, _tmp40)
        tmp42 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp44 = _tmp43 + tmp42
        _tmp43 = tl.where(rmask, tmp44, _tmp43)
        tmp46 = tl.broadcast_to(tmp45, [XBLOCK, RBLOCK])
        tmp48 = _tmp47 + tmp46
        _tmp47 = tl.where(rmask, tmp48, _tmp47)
        tmp50 = tmp45 + tmp49
        tmp52 = tmp50 + tmp51
        tmp54 = tmp52 + tmp53
        tmp56 = tmp54 * tmp55
        tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
        tmp59 = _tmp58 + tmp57
        _tmp58 = tl.where(rmask, tmp59, _tmp58)
        tmp60 = tl.broadcast_to(tmp54, [XBLOCK, RBLOCK])
        tmp62 = _tmp61 + tmp60
        _tmp61 = tl.where(rmask, tmp62, _tmp61)
        tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
        tmp66 = _tmp65 + tmp64
        _tmp65 = tl.where(rmask, tmp66, _tmp65)
        tmp68 = tmp63 + tmp67
        tmp70 = tmp68 + tmp69
        tmp72 = tmp70 + tmp71
        tmp74 = tmp72 * tmp73
        tmp75 = tl.broadcast_to(tmp74, [XBLOCK, RBLOCK])
        tmp77 = _tmp76 + tmp75
        _tmp76 = tl.where(rmask, tmp77, _tmp76)
        tmp78 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
        tmp80 = _tmp79 + tmp78
        _tmp79 = tl.where(rmask, tmp80, _tmp79)
        tmp82 = tl.broadcast_to(tmp81, [XBLOCK, RBLOCK])
        tmp84 = _tmp83 + tmp82
        _tmp83 = tl.where(rmask, tmp84, _tmp83)
        tmp86 = tmp81 + tmp85
        tmp88 = tmp86 + tmp87
        tmp90 = tmp88 + tmp89
        tmp92 = tmp90 * tmp91
        tmp93 = tl.broadcast_to(tmp92, [XBLOCK, RBLOCK])
        tmp95 = _tmp94 + tmp93
        _tmp94 = tl.where(rmask, tmp95, _tmp94)
        tmp96 = tl.broadcast_to(tmp90, [XBLOCK, RBLOCK])
        tmp98 = _tmp97 + tmp96
        _tmp97 = tl.where(rmask, tmp98, _tmp97)
        tmp100 = tl.broadcast_to(tmp99, [XBLOCK, RBLOCK])
        tmp102 = _tmp101 + tmp100
        _tmp101 = tl.where(rmask, tmp102, _tmp101)
        tmp104 = tmp99 + tmp103
        tmp106 = tmp104 + tmp105
        tmp108 = tmp106 + tmp107
        tmp110 = tmp108 * tmp109
        tmp111 = tl.broadcast_to(tmp110, [XBLOCK, RBLOCK])
        tmp113 = _tmp112 + tmp111
        _tmp112 = tl.where(rmask, tmp113, _tmp112)
        tmp114 = tl.broadcast_to(tmp108, [XBLOCK, RBLOCK])
        tmp116 = _tmp115 + tmp114
        _tmp115 = tl.where(rmask, tmp116, _tmp115)
        tmp118 = tl.broadcast_to(tmp117, [XBLOCK, RBLOCK])
        tmp120 = _tmp119 + tmp118
        _tmp119 = tl.where(rmask, tmp120, _tmp119)
        tmp122 = tmp117 + tmp121
        tmp124 = tmp122 + tmp123
        tmp126 = tmp124 + tmp125
        tmp128 = tmp126 * tmp127
        tmp129 = tl.broadcast_to(tmp128, [XBLOCK, RBLOCK])
        tmp131 = _tmp130 + tmp129
        _tmp130 = tl.where(rmask, tmp131, _tmp130)
        tmp132 = tl.broadcast_to(tmp126, [XBLOCK, RBLOCK])
        tmp134 = _tmp133 + tmp132
        _tmp133 = tl.where(rmask, tmp134, _tmp133)
        tmp136 = tl.broadcast_to(tmp135, [XBLOCK, RBLOCK])
        tmp138 = _tmp137 + tmp136
        _tmp137 = tl.where(rmask, tmp138, _tmp137)
        tmp140 = tmp135 + tmp139
        tmp142 = tmp140 + tmp141
        tmp144 = tmp142 + tmp143
        tmp146 = tmp144 * tmp145
        tmp147 = tl.broadcast_to(tmp146, [XBLOCK, RBLOCK])
        tmp149 = _tmp148 + tmp147
        _tmp148 = tl.where(rmask, tmp149, _tmp148)
        tmp150 = tl.broadcast_to(tmp144, [XBLOCK, RBLOCK])
        tmp152 = _tmp151 + tmp150
        _tmp151 = tl.where(rmask, tmp152, _tmp151)
        tmp154 = tl.broadcast_to(tmp153, [XBLOCK, RBLOCK])
        tmp156 = _tmp155 + tmp154
        _tmp155 = tl.where(rmask, tmp156, _tmp155)
        tmp158 = tmp153 + tmp157
        tmp160 = tmp158 + tmp159
        tmp162 = tmp160 + tmp161
        tmp164 = tmp162 * tmp163
        tmp165 = tl.broadcast_to(tmp164, [XBLOCK, RBLOCK])
        tmp167 = _tmp166 + tmp165
        _tmp166 = tl.where(rmask, tmp167, _tmp166)
        tmp168 = tl.broadcast_to(tmp162, [XBLOCK, RBLOCK])
        tmp170 = _tmp169 + tmp168
        _tmp169 = tl.where(rmask, tmp170, _tmp169)
        tmp172 = tl.broadcast_to(tmp171, [XBLOCK, RBLOCK])
        tmp174 = _tmp173 + tmp172
        _tmp173 = tl.where(rmask, tmp174, _tmp173)
        tmp176 = tmp171 + tmp175
        tmp178 = tmp176 + tmp177
        tmp180 = tmp178 + tmp179
        tmp182 = tmp180 * tmp181
        tmp183 = tl.broadcast_to(tmp182, [XBLOCK, RBLOCK])
        tmp185 = _tmp184 + tmp183
        _tmp184 = tl.where(rmask, tmp185, _tmp184)
        tmp186 = tl.broadcast_to(tmp180, [XBLOCK, RBLOCK])
        tmp188 = _tmp187 + tmp186
        _tmp187 = tl.where(rmask, tmp188, _tmp187)
        tmp190 = tl.broadcast_to(tmp189, [XBLOCK, RBLOCK])
        tmp192 = _tmp191 + tmp190
        _tmp191 = tl.where(rmask, tmp192, _tmp191)
        tmp194 = tl.broadcast_to(tmp193, [XBLOCK, RBLOCK])
        tmp196 = _tmp195 + tmp194
        _tmp195 = tl.where(rmask, tmp196, _tmp195)
        tmp198 = tmp193 + tmp197
        tmp200 = tmp198 + tmp199
        tmp202 = tmp200 + tmp201
        tmp204 = tmp202 * tmp203
        tmp205 = tl.broadcast_to(tmp204, [XBLOCK, RBLOCK])
        tmp207 = _tmp206 + tmp205
        _tmp206 = tl.where(rmask, tmp207, _tmp206)
        tmp208 = tl.broadcast_to(tmp202, [XBLOCK, RBLOCK])
        tmp210 = _tmp209 + tmp208
        _tmp209 = tl.where(rmask, tmp210, _tmp209)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tmp40 = tl.sum(_tmp40, 1)[:, None]
    tmp43 = tl.sum(_tmp43, 1)[:, None]
    tmp47 = tl.sum(_tmp47, 1)[:, None]
    tmp58 = tl.sum(_tmp58, 1)[:, None]
    tmp61 = tl.sum(_tmp61, 1)[:, None]
    tmp65 = tl.sum(_tmp65, 1)[:, None]
    tmp76 = tl.sum(_tmp76, 1)[:, None]
    tmp79 = tl.sum(_tmp79, 1)[:, None]
    tmp83 = tl.sum(_tmp83, 1)[:, None]
    tmp94 = tl.sum(_tmp94, 1)[:, None]
    tmp97 = tl.sum(_tmp97, 1)[:, None]
    tmp101 = tl.sum(_tmp101, 1)[:, None]
    tmp112 = tl.sum(_tmp112, 1)[:, None]
    tmp115 = tl.sum(_tmp115, 1)[:, None]
    tmp119 = tl.sum(_tmp119, 1)[:, None]
    tmp130 = tl.sum(_tmp130, 1)[:, None]
    tmp133 = tl.sum(_tmp133, 1)[:, None]
    tmp137 = tl.sum(_tmp137, 1)[:, None]
    tmp148 = tl.sum(_tmp148, 1)[:, None]
    tmp151 = tl.sum(_tmp151, 1)[:, None]
    tmp155 = tl.sum(_tmp155, 1)[:, None]
    tmp166 = tl.sum(_tmp166, 1)[:, None]
    tmp169 = tl.sum(_tmp169, 1)[:, None]
    tmp173 = tl.sum(_tmp173, 1)[:, None]
    tmp184 = tl.sum(_tmp184, 1)[:, None]
    tmp187 = tl.sum(_tmp187, 1)[:, None]
    tmp191 = tl.sum(_tmp191, 1)[:, None]
    tmp195 = tl.sum(_tmp195, 1)[:, None]
    tmp206 = tl.sum(_tmp206, 1)[:, None]
    tmp209 = tl.sum(_tmp209, 1)[:, None]
    tmp211 = tmp11 + tmp29
    tmp212 = tmp211 + tmp47
    tmp213 = tmp212 + tmp65
    tmp214 = tmp213 + tmp101
    tmp215 = tmp214 + tmp83
    tmp216 = tmp215 + tmp137
    tmp217 = tmp216 + tmp173
    tmp218 = tmp217 + tmp195
    tmp219 = tmp218 + tmp119
    tmp220 = tmp219 + tmp155
    tmp221 = tmp220 + tmp191
    tmp222 = tmp4 + tmp22
    tmp223 = tmp222 + tmp40
    tmp224 = tmp223 + tmp58
    tmp225 = tmp224 + tmp76
    tmp226 = tmp225 + tmp112
    tmp227 = tmp226 + tmp94
    tmp228 = tmp227 + tmp148
    tmp229 = tmp228 + tmp184
    tmp230 = tmp229 + tmp206
    tmp231 = tmp230 + tmp130
    tmp232 = tmp231 + tmp166
    tmp233 = tmp7 + tmp25
    tmp234 = tmp233 + tmp43
    tmp235 = tmp234 + tmp61
    tmp236 = tmp235 + tmp79
    tmp237 = tmp236 + tmp115
    tmp238 = tmp237 + tmp97
    tmp239 = tmp238 + tmp151
    tmp240 = tmp239 + tmp187
    tmp241 = tmp240 + tmp209
    tmp242 = tmp241 + tmp133
    tmp243 = tmp242 + tmp169
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp221, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp232, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp243, None)
''')
