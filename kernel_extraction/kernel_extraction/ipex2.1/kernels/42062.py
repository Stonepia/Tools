

# Original file: ./AlbertForQuestionAnswering__0_backward_207.1/AlbertForQuestionAnswering__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/or/corahcwlbkmeqzdbfcw2vmnwvh2qqovafn23zies7qr6lym6pk6i.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.sum]

triton_red_fused_add_native_layer_norm_backward_sum_15 = async_compile.triton('triton_red_fused_add_native_layer_norm_backward_sum_15', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: 'i32', 40: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_sum_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_backward_sum_15(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp37 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp40 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp44 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp51 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp54 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp58 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp65 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp68 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp72 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp79 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp82 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp86 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp93 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp96 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp100 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp107 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp110 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp114 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp121 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp124 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp128 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp135 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp138 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp142 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp149 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp152 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp156 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp163 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp166 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr3 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr4 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr5 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr6 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.load(in_ptr7 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp34 = tl.load(in_ptr8 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp42 = tl.load(in_ptr9 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp46 = tl.load(in_ptr10 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp48 = tl.load(in_ptr11 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp56 = tl.load(in_ptr12 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp60 = tl.load(in_ptr13 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp62 = tl.load(in_ptr14 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp70 = tl.load(in_ptr15 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp74 = tl.load(in_ptr16 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp76 = tl.load(in_ptr17 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp84 = tl.load(in_ptr18 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp88 = tl.load(in_ptr19 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp90 = tl.load(in_ptr20 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp98 = tl.load(in_ptr21 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp102 = tl.load(in_ptr22 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp104 = tl.load(in_ptr23 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp112 = tl.load(in_ptr24 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp116 = tl.load(in_ptr25 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp118 = tl.load(in_ptr26 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp126 = tl.load(in_ptr27 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp130 = tl.load(in_ptr28 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp132 = tl.load(in_ptr29 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp140 = tl.load(in_ptr30 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp144 = tl.load(in_ptr31 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp146 = tl.load(in_ptr32 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp154 = tl.load(in_ptr33 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp158 = tl.load(in_ptr34 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp160 = tl.load(in_ptr35 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
        tmp11 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
        tmp19 = tmp14 + tmp18
        tmp21 = tmp19 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask, tmp24, _tmp23)
        tmp25 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
        tmp33 = tmp28 + tmp32
        tmp35 = tmp33 * tmp34
        tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
        tmp38 = _tmp37 + tmp36
        _tmp37 = tl.where(rmask, tmp38, _tmp37)
        tmp39 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
        tmp41 = _tmp40 + tmp39
        _tmp40 = tl.where(rmask, tmp41, _tmp40)
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
        tmp45 = _tmp44 + tmp43
        _tmp44 = tl.where(rmask, tmp45, _tmp44)
        tmp47 = tmp42 + tmp46
        tmp49 = tmp47 * tmp48
        tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
        tmp52 = _tmp51 + tmp50
        _tmp51 = tl.where(rmask, tmp52, _tmp51)
        tmp53 = tl.broadcast_to(tmp47, [XBLOCK, RBLOCK])
        tmp55 = _tmp54 + tmp53
        _tmp54 = tl.where(rmask, tmp55, _tmp54)
        tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
        tmp59 = _tmp58 + tmp57
        _tmp58 = tl.where(rmask, tmp59, _tmp58)
        tmp61 = tmp56 + tmp60
        tmp63 = tmp61 * tmp62
        tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
        tmp66 = _tmp65 + tmp64
        _tmp65 = tl.where(rmask, tmp66, _tmp65)
        tmp67 = tl.broadcast_to(tmp61, [XBLOCK, RBLOCK])
        tmp69 = _tmp68 + tmp67
        _tmp68 = tl.where(rmask, tmp69, _tmp68)
        tmp71 = tl.broadcast_to(tmp70, [XBLOCK, RBLOCK])
        tmp73 = _tmp72 + tmp71
        _tmp72 = tl.where(rmask, tmp73, _tmp72)
        tmp75 = tmp70 + tmp74
        tmp77 = tmp75 * tmp76
        tmp78 = tl.broadcast_to(tmp77, [XBLOCK, RBLOCK])
        tmp80 = _tmp79 + tmp78
        _tmp79 = tl.where(rmask, tmp80, _tmp79)
        tmp81 = tl.broadcast_to(tmp75, [XBLOCK, RBLOCK])
        tmp83 = _tmp82 + tmp81
        _tmp82 = tl.where(rmask, tmp83, _tmp82)
        tmp85 = tl.broadcast_to(tmp84, [XBLOCK, RBLOCK])
        tmp87 = _tmp86 + tmp85
        _tmp86 = tl.where(rmask, tmp87, _tmp86)
        tmp89 = tmp84 + tmp88
        tmp91 = tmp89 * tmp90
        tmp92 = tl.broadcast_to(tmp91, [XBLOCK, RBLOCK])
        tmp94 = _tmp93 + tmp92
        _tmp93 = tl.where(rmask, tmp94, _tmp93)
        tmp95 = tl.broadcast_to(tmp89, [XBLOCK, RBLOCK])
        tmp97 = _tmp96 + tmp95
        _tmp96 = tl.where(rmask, tmp97, _tmp96)
        tmp99 = tl.broadcast_to(tmp98, [XBLOCK, RBLOCK])
        tmp101 = _tmp100 + tmp99
        _tmp100 = tl.where(rmask, tmp101, _tmp100)
        tmp103 = tmp98 + tmp102
        tmp105 = tmp103 * tmp104
        tmp106 = tl.broadcast_to(tmp105, [XBLOCK, RBLOCK])
        tmp108 = _tmp107 + tmp106
        _tmp107 = tl.where(rmask, tmp108, _tmp107)
        tmp109 = tl.broadcast_to(tmp103, [XBLOCK, RBLOCK])
        tmp111 = _tmp110 + tmp109
        _tmp110 = tl.where(rmask, tmp111, _tmp110)
        tmp113 = tl.broadcast_to(tmp112, [XBLOCK, RBLOCK])
        tmp115 = _tmp114 + tmp113
        _tmp114 = tl.where(rmask, tmp115, _tmp114)
        tmp117 = tmp112 + tmp116
        tmp119 = tmp117 * tmp118
        tmp120 = tl.broadcast_to(tmp119, [XBLOCK, RBLOCK])
        tmp122 = _tmp121 + tmp120
        _tmp121 = tl.where(rmask, tmp122, _tmp121)
        tmp123 = tl.broadcast_to(tmp117, [XBLOCK, RBLOCK])
        tmp125 = _tmp124 + tmp123
        _tmp124 = tl.where(rmask, tmp125, _tmp124)
        tmp127 = tl.broadcast_to(tmp126, [XBLOCK, RBLOCK])
        tmp129 = _tmp128 + tmp127
        _tmp128 = tl.where(rmask, tmp129, _tmp128)
        tmp131 = tmp126 + tmp130
        tmp133 = tmp131 * tmp132
        tmp134 = tl.broadcast_to(tmp133, [XBLOCK, RBLOCK])
        tmp136 = _tmp135 + tmp134
        _tmp135 = tl.where(rmask, tmp136, _tmp135)
        tmp137 = tl.broadcast_to(tmp131, [XBLOCK, RBLOCK])
        tmp139 = _tmp138 + tmp137
        _tmp138 = tl.where(rmask, tmp139, _tmp138)
        tmp141 = tl.broadcast_to(tmp140, [XBLOCK, RBLOCK])
        tmp143 = _tmp142 + tmp141
        _tmp142 = tl.where(rmask, tmp143, _tmp142)
        tmp145 = tmp140 + tmp144
        tmp147 = tmp145 * tmp146
        tmp148 = tl.broadcast_to(tmp147, [XBLOCK, RBLOCK])
        tmp150 = _tmp149 + tmp148
        _tmp149 = tl.where(rmask, tmp150, _tmp149)
        tmp151 = tl.broadcast_to(tmp145, [XBLOCK, RBLOCK])
        tmp153 = _tmp152 + tmp151
        _tmp152 = tl.where(rmask, tmp153, _tmp152)
        tmp155 = tl.broadcast_to(tmp154, [XBLOCK, RBLOCK])
        tmp157 = _tmp156 + tmp155
        _tmp156 = tl.where(rmask, tmp157, _tmp156)
        tmp159 = tmp154 + tmp158
        tmp161 = tmp159 * tmp160
        tmp162 = tl.broadcast_to(tmp161, [XBLOCK, RBLOCK])
        tmp164 = _tmp163 + tmp162
        _tmp163 = tl.where(rmask, tmp164, _tmp163)
        tmp165 = tl.broadcast_to(tmp159, [XBLOCK, RBLOCK])
        tmp167 = _tmp166 + tmp165
        _tmp166 = tl.where(rmask, tmp167, _tmp166)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tmp37 = tl.sum(_tmp37, 1)[:, None]
    tmp40 = tl.sum(_tmp40, 1)[:, None]
    tmp44 = tl.sum(_tmp44, 1)[:, None]
    tmp51 = tl.sum(_tmp51, 1)[:, None]
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    tmp58 = tl.sum(_tmp58, 1)[:, None]
    tmp65 = tl.sum(_tmp65, 1)[:, None]
    tmp68 = tl.sum(_tmp68, 1)[:, None]
    tmp72 = tl.sum(_tmp72, 1)[:, None]
    tmp79 = tl.sum(_tmp79, 1)[:, None]
    tmp82 = tl.sum(_tmp82, 1)[:, None]
    tmp86 = tl.sum(_tmp86, 1)[:, None]
    tmp93 = tl.sum(_tmp93, 1)[:, None]
    tmp96 = tl.sum(_tmp96, 1)[:, None]
    tmp100 = tl.sum(_tmp100, 1)[:, None]
    tmp107 = tl.sum(_tmp107, 1)[:, None]
    tmp110 = tl.sum(_tmp110, 1)[:, None]
    tmp114 = tl.sum(_tmp114, 1)[:, None]
    tmp121 = tl.sum(_tmp121, 1)[:, None]
    tmp124 = tl.sum(_tmp124, 1)[:, None]
    tmp128 = tl.sum(_tmp128, 1)[:, None]
    tmp135 = tl.sum(_tmp135, 1)[:, None]
    tmp138 = tl.sum(_tmp138, 1)[:, None]
    tmp142 = tl.sum(_tmp142, 1)[:, None]
    tmp149 = tl.sum(_tmp149, 1)[:, None]
    tmp152 = tl.sum(_tmp152, 1)[:, None]
    tmp156 = tl.sum(_tmp156, 1)[:, None]
    tmp163 = tl.sum(_tmp163, 1)[:, None]
    tmp166 = tl.sum(_tmp166, 1)[:, None]
    tmp168 = tmp2 + tmp16
    tmp169 = tmp168 + tmp30
    tmp170 = tmp169 + tmp44
    tmp171 = tmp170 + tmp58
    tmp172 = tmp171 + tmp72
    tmp173 = tmp172 + tmp100
    tmp174 = tmp173 + tmp128
    tmp175 = tmp174 + tmp156
    tmp176 = tmp175 + tmp86
    tmp177 = tmp176 + tmp114
    tmp178 = tmp177 + tmp142
    tmp179 = tmp9 + tmp23
    tmp180 = tmp179 + tmp37
    tmp181 = tmp180 + tmp51
    tmp182 = tmp181 + tmp65
    tmp183 = tmp182 + tmp79
    tmp184 = tmp183 + tmp107
    tmp185 = tmp184 + tmp135
    tmp186 = tmp185 + tmp163
    tmp187 = tmp186 + tmp93
    tmp188 = tmp187 + tmp121
    tmp189 = tmp188 + tmp149
    tmp190 = tmp12 + tmp26
    tmp191 = tmp190 + tmp40
    tmp192 = tmp191 + tmp54
    tmp193 = tmp192 + tmp68
    tmp194 = tmp193 + tmp82
    tmp195 = tmp194 + tmp110
    tmp196 = tmp195 + tmp138
    tmp197 = tmp196 + tmp166
    tmp198 = tmp197 + tmp96
    tmp199 = tmp198 + tmp124
    tmp200 = tmp199 + tmp152
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp178, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp189, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp200, None)
''')
