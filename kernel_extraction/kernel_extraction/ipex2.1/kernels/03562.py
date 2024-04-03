

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/wx/cwxbwclnnzmw7rp2gajaqdaljwfgtnkertbp256et54nebdfthrz.py
# Source Nodes: [add_32, add_35, add_37, add_39, add_41, add_43, add_45, add_47, add_49, add_51, add_53, add_55, add_57, add_59, add_61, add_63, add_65, add_67, l__self___decoder_dropout], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout, aten.native_dropout_backward, aten.pow, aten.sum]
# add_32 => add_40
# add_35 => add_44
# add_37 => add_46
# add_39 => add_49
# add_41 => add_52
# add_43 => add_54
# add_45 => add_57
# add_47 => add_60
# add_49 => add_62
# add_51 => add_65
# add_53 => add_68
# add_55 => add_70
# add_57 => add_73
# add_59 => add_76
# add_61 => add_78
# add_63 => add_81
# add_65 => add_84
# add_67 => add_86
# l__self___decoder_dropout => mul_84, mul_85
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
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*i1', 19: '*fp32', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp32', 24: '*i1', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp16', 31: 'i32', 32: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_native_dropout_backward_pow_sum_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr5, out_ptr6, xnumel, rnumel):
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
    tmp39 = tl.load(in_ptr13 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp42 = tl.load(in_ptr14 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp45 = tl.load(in_ptr15 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp48 = tl.load(in_ptr16 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp51 = tl.load(in_ptr17 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp55 = tl.load(in_ptr18 + (r1 + (512*x0)), rmask)
    tmp59 = tl.load(in_ptr19 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.load(in_ptr20 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp64 = tl.load(in_ptr21 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp67 = tl.load(in_ptr22 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp75 = tl.load(in_ptr23 + (x0), None, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr24 + (r1 + (512*x0)), rmask)
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
    tmp53 = 0.04419417382415922
    tmp54 = tmp52 * tmp53
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp56 * tmp4
    tmp58 = tmp54 * tmp57
    tmp60 = tmp58 * tmp59
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp50 + tmp62
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp63 + tmp65
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp66 + tmp68
    tmp70 = tmp60 * tmp69
    tmp71 = tl.broadcast_to(tmp70, [RBLOCK])
    tmp73 = tl.where(rmask, tmp71, 0)
    tmp74 = triton_helpers.promote_to_tensor(tl.sum(tmp73, 0))
    tmp76 = tmp60 * tmp75
    tmp77 = -0.5
    tmp78 = tmp74 * tmp77
    tmp79 = tmp75 * tmp75
    tmp80 = tmp79 * tmp75
    tmp81 = tmp78 * tmp80
    tmp82 = 512.0
    tmp83 = tmp81 / tmp82
    tmp84 = 2.0
    tmp85 = tmp69 * tmp84
    tmp86 = tmp83 * tmp85
    tmp87 = tmp76 + tmp86
    tmp88 = tmp87.to(tl.float32)
    tmp90 = tmp89.to(tl.float32)
    tmp91 = tmp90 * tmp4
    tmp92 = tmp88 * tmp91
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp14, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp26, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp38, rmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp50, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp87, rmask)
    tl.store(out_ptr6 + (r1 + (512*x0)), tmp92, rmask)
''')
