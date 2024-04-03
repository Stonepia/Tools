

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/lz/clzqxpwon7azl6djxbxornw4pbcdei3u6u7dvggoo65dk7atlnpi.py
# Source Nodes: [add_103, add_105, add_107, add_111, add_113, add_115, add_119, add_56, add_59, add_63, add_65, add_67, add_71, add_73, add_75, add_79, add_81, add_83, add_87, add_89, add_91, add_95, add_97, add_99, l__self___decoder_dropout], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.native_dropout_backward, aten.pow, aten.sum]
# add_103 => add_124
# add_105 => add_127
# add_107 => add_130
# add_111 => add_134
# add_113 => add_137
# add_115 => add_140
# add_119 => add_144
# add_56 => add_66
# add_59 => add_70
# add_63 => add_74
# add_65 => add_77
# add_67 => add_80
# add_71 => add_84
# add_73 => add_87
# add_75 => add_90
# add_79 => add_94
# add_81 => add_97
# add_83 => add_100
# add_87 => add_104
# add_89 => add_107
# add_91 => add_110
# add_95 => add_114
# add_97 => add_117
# add_99 => add_120
# l__self___decoder_dropout => mul_148, mul_149
triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_3 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_3', '''
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*bf16', 12: '*bf16', 13: '*bf16', 14: '*bf16', 15: '*bf16', 16: '*bf16', 17: '*bf16', 18: '*bf16', 19: '*bf16', 20: '*bf16', 21: '*bf16', 22: '*bf16', 23: '*bf16', 24: '*bf16', 25: '*bf16', 26: '*i1', 27: '*fp32', 28: '*bf16', 29: '*fp32', 30: '*i1', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*bf16', 39: 'i32', 40: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr7, out_ptr8, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask)
    tmp2 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr12 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr13 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp42 = tl.load(in_ptr14 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp45 = tl.load(in_ptr15 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp48 = tl.load(in_ptr16 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp51 = tl.load(in_ptr17 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp54 = tl.load(in_ptr18 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp57 = tl.load(in_ptr19 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp60 = tl.load(in_ptr20 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp63 = tl.load(in_ptr21 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp66 = tl.load(in_ptr22 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp69 = tl.load(in_ptr23 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp72 = tl.load(in_ptr24 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp75 = tl.load(in_ptr25 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp77 = tl.load(in_ptr26 + (r1 + (512*x0)), rmask)
    tmp81 = tl.load(in_ptr27 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tl.load(in_ptr28 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp91 = tl.load(in_ptr29 + (x0), None, eviction_policy='evict_last')
    tmp105 = tl.load(in_ptr30 + (r1 + (512*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 + tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 + tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 + tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 + tmp28
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 + tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp38 + tmp40
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 + tmp43
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 + tmp46
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp47 + tmp49
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp50 + tmp52
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp53 + tmp55
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp56 + tmp58
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp59 + tmp61
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp62 + tmp64
    tmp67 = tmp66.to(tl.float32)
    tmp68 = tmp65 + tmp67
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp68 + tmp70
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp71 + tmp73
    tmp76 = tmp75.to(tl.float32)
    tmp78 = tmp77.to(tl.float32)
    tmp79 = tmp78 * tmp4
    tmp80 = tmp76 * tmp79
    tmp82 = tmp80 * tmp81
    tmp84 = tmp83.to(tl.float32)
    tmp85 = tmp74 + tmp84
    tmp86 = tmp82 * tmp85
    tmp87 = tl.broadcast_to(tmp86, [RBLOCK])
    tmp89 = tl.where(rmask, tmp87, 0)
    tmp90 = triton_helpers.promote_to_tensor(tl.sum(tmp89, 0))
    tmp92 = tmp82 * tmp91
    tmp93 = -0.5
    tmp94 = tmp90 * tmp93
    tmp95 = tmp91 * tmp91
    tmp96 = tmp95 * tmp91
    tmp97 = tmp94 * tmp96
    tmp98 = 512.0
    tmp99 = tmp97 / tmp98
    tmp100 = 2.0
    tmp101 = tmp85 * tmp100
    tmp102 = tmp99 * tmp101
    tmp103 = tmp92 + tmp102
    tmp104 = tmp103.to(tl.float32)
    tmp106 = tmp105.to(tl.float32)
    tmp107 = tmp106 * tmp4
    tmp108 = tmp104 * tmp107
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp14, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp26, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp38, rmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp50, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp62, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp74, rmask)
    tl.store(out_ptr7 + (r1 + (512*x0)), tmp103, rmask)
    tl.store(out_ptr8 + (r1 + (512*x0)), tmp108, rmask)
''')
