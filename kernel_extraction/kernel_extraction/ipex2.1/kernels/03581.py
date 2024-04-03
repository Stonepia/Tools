

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/2f/c2frqaifjnunfpzzav33r7k7uaspfr43wc5vg42tebyttepqhefc.py
# Source Nodes: [add_10, add_12, add_14, add_16, add_18, add_20, add_22, add_24, add_26, add_4, add_6, add_8, l__self___encoder_dropout], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.native_dropout_backward, aten.pow, aten.sum]
# add_10 => add_13
# add_12 => add_16
# add_14 => add_18
# add_16 => add_21
# add_18 => add_23
# add_20 => add_26
# add_22 => add_28
# add_24 => add_31
# add_26 => add_33
# add_4 => add_6
# add_6 => add_8
# add_8 => add_11
# l__self___encoder_dropout => mul_1, mul_2
triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_22 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_22', '''
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
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp32', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*i1', 28: '*fp32', 29: '*fp16', 30: '*fp32', 31: '*i1', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp16', 37: 'i32', 38: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_22', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 4096
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
    tmp39 = tl.load(in_ptr13 + (r1 + (512*x0)), rmask, other=0.0)
    tmp40 = tl.load(in_ptr14 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr15 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp46 = tl.load(in_ptr16 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp49 = tl.load(in_ptr17 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp52 = tl.load(in_ptr18 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp55 = tl.load(in_ptr19 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp58 = tl.load(in_ptr20 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp61 = tl.load(in_ptr21 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp64 = tl.load(in_ptr22 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp67 = tl.load(in_ptr23 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp70 = tl.load(in_ptr24 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp73 = tl.load(in_ptr25 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp76 = tl.load(in_ptr26 + (r1 + (512*x0)), rmask)
    tmp80 = tl.load(in_ptr27 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp82 = tl.load(in_ptr28 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp90 = tl.load(in_ptr29 + (x0), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr30 + (r1 + (512*x0)), rmask)
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
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp39 + tmp41
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp42 + tmp44
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp45 + tmp47
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp48 + tmp50
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp51 + tmp53
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp54 + tmp56
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp57 + tmp59
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp60 + tmp62
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp63 + tmp65
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp66 + tmp68
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp69 + tmp71
    tmp74 = tmp73.to(tl.float32)
    tmp75 = tmp72 + tmp74
    tmp77 = tmp76.to(tl.float32)
    tmp78 = tmp77 * tmp4
    tmp79 = tmp75 * tmp78
    tmp81 = tmp79 * tmp80
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp38 + tmp83
    tmp85 = tmp81 * tmp84
    tmp86 = tl.broadcast_to(tmp85, [RBLOCK])
    tmp88 = tl.where(rmask, tmp86, 0)
    tmp89 = triton_helpers.promote_to_tensor(tl.sum(tmp88, 0))
    tmp91 = tmp81 * tmp90
    tmp92 = -0.5
    tmp93 = tmp89 * tmp92
    tmp94 = tmp90 * tmp90
    tmp95 = tmp94 * tmp90
    tmp96 = tmp93 * tmp95
    tmp97 = 512.0
    tmp98 = tmp96 / tmp97
    tmp99 = 2.0
    tmp100 = tmp84 * tmp99
    tmp101 = tmp98 * tmp100
    tmp102 = tmp91 + tmp101
    tmp103 = tmp102.to(tl.float32)
    tmp105 = tmp104.to(tl.float32)
    tmp106 = tmp105 * tmp4
    tmp107 = tmp103 * tmp106
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp14, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp26, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp38, rmask)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp79, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp102, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp107, rmask)
''')
