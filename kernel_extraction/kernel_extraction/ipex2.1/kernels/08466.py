

# Original file: ./DistillGPT2__0_backward_99.1/DistillGPT2__0_backward_99.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/li/clian3shvn73cg2jmvo57ziz5525m22lwas2sfc7fhqibn7bx5lo.py
# Source Nodes: [add_1, add_12, add_13, add_16, add_17, add_20, add_21, add_24, add_4, add_5, add_8, add_9, l__self___transformer_h_1_ln_1, l__self___transformer_h_1_ln_2, l__self___transformer_h_3_ln_1, l__self___transformer_h_3_ln_2, l__self___transformer_h_5_ln_1, l__self___transformer_h_5_ln_2, l__self___transformer_ln_f], Original ATen: [aten._to_copy, aten.add, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_1 => add_3
# add_12 => add_24
# add_13 => add_27
# add_16 => add_32
# add_17 => add_35
# add_20 => add_40
# add_21 => add_43
# add_24 => add_48
# add_4 => add_8
# add_5 => add_11
# add_8 => add_16
# add_9 => add_19
# l__self___transformer_h_1_ln_1 => mul_16, sub_3
# l__self___transformer_h_1_ln_2 => mul_22, sub_5
# l__self___transformer_h_3_ln_1 => mul_44, sub_9
# l__self___transformer_h_3_ln_2 => mul_50, sub_11
# l__self___transformer_h_5_ln_1 => mul_72, sub_15
# l__self___transformer_h_5_ln_2 => mul_78, sub_17
# l__self___transformer_ln_f => mul_86, sub_18
triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_4 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_4', '''
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
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp32', 12: '*fp32', 13: '*fp16', 14: '*fp32', 15: '*fp32', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp32', 20: '*fp32', 21: '*fp16', 22: '*fp32', 23: '*fp32', 24: '*fp16', 25: '*fp32', 26: '*fp32', 27: '*fp16', 28: '*fp32', 29: '*i1', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp16', 41: 'i32', 42: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr11, out_ptr12, xnumel, rnumel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr9 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr10 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr13 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr15 + (x0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr16 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp41 = tl.load(in_ptr17 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp44 = tl.load(in_ptr18 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp47 = tl.load(in_ptr19 + (x0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr20 + (x0), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr21 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp54 = tl.load(in_ptr22 + (x0), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr23 + (x0), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr24 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp61 = tl.load(in_ptr25 + (x0), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr26 + (x0), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr27 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp67 = tl.load(in_ptr28 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tl.load(in_ptr29 + (r1 + (768*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 + tmp6
    tmp15 = tmp13 - tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp13 + tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22 + tmp20
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 + tmp25
    tmp28 = tmp26 - tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 + tmp26
    tmp35 = tmp33 - tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp33 + tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 + tmp40
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp43 + tmp45
    tmp48 = tmp46 - tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp52 + tmp46
    tmp55 = tmp53 - tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp53 + tmp59
    tmp62 = tmp60 - tmp61
    tmp64 = tmp62 * tmp63
    tmp66 = tmp65.to(tl.float32)
    tmp68 = tmp66 * tmp67
    tmp69 = tl.broadcast_to(tmp68, [RBLOCK])
    tmp71 = tl.where(rmask, tmp69, 0)
    tmp72 = triton_helpers.promote_to_tensor(tl.sum(tmp71, 0))
    tmp73 = tmp68 * tmp64
    tmp74 = tl.broadcast_to(tmp73, [RBLOCK])
    tmp76 = tl.where(rmask, tmp74, 0)
    tmp77 = triton_helpers.promote_to_tensor(tl.sum(tmp76, 0))
    tmp78 = 768.0
    tmp79 = tmp63 / tmp78
    tmp80 = tmp68 * tmp78
    tmp81 = tmp80 - tmp72
    tmp82 = tmp64 * tmp77
    tmp83 = tmp81 - tmp82
    tmp84 = tmp79 * tmp83
    tmp85 = tmp84.to(tl.float32)
    tmp87 = tmp86.to(tl.float32)
    tmp88 = 1.1111111111111112
    tmp89 = tmp87 * tmp88
    tmp90 = tmp85 * tmp89
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp10, rmask)
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp17, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp20, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp30, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp37, rmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp40, rmask)
    tl.store(out_ptr6 + (r1 + (768*x0)), tmp50, rmask)
    tl.store(out_ptr7 + (r1 + (768*x0)), tmp57, rmask)
    tl.store(out_ptr8 + (r1 + (768*x0)), tmp64, rmask)
    tl.store(out_ptr11 + (r1 + (768*x0)), tmp84, rmask)
    tl.store(out_ptr12 + (r1 + (768*x0)), tmp90, rmask)
''')
