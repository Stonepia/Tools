

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/fl/cflwhjcaagrwehucylou3leger6tbmzouvdum2jcypel3db3bdxs.py
# Source Nodes: [cross_entropy, to_35], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum, aten.where]
# cross_entropy => full_default_87
# to_35 => convert_element_type_84
triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_21 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_21', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*i1', 18: '*fp16', 19: '*fp16', 20: '*i1', 21: '*fp32', 22: '*i1', 23: '*fp16', 24: '*fp16', 25: 'i32', 26: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_dropout_backward_nll_loss_forward_pow_sum_where_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr6 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr12 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr13 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr14 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr15 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr16 + (r1 + (512*x0)), rmask)
    tmp38 = tl.load(in_ptr17 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = tl.load(in_ptr18 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp48 = tl.load(in_ptr19 + (r1 + (512*x0)), rmask)
    tmp49 = tl.load(in_ptr20 + (x0), None, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr21 + (r1 + (512*x0)), rmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 + tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = 1.1111111111111112
    tmp36 = tmp34 * tmp35
    tmp37 = tmp32 * tmp36
    tmp39 = tmp37 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 * tmp42
    tmp44 = tl.broadcast_to(tmp43, [RBLOCK])
    tmp46 = tl.where(rmask, tmp44, 0)
    tmp47 = triton_helpers.promote_to_tensor(tl.sum(tmp46, 0))
    tmp50 = tmp40 * tmp49
    tmp51 = tmp50.to(tl.float32)
    tmp52 = -0.5
    tmp53 = tmp47 * tmp52
    tmp54 = tmp49 * tmp49
    tmp55 = tmp54 * tmp49
    tmp56 = tmp53 * tmp55
    tmp57 = 512.0
    tmp58 = tmp56 / tmp57
    tmp59 = 2.0
    tmp60 = tmp42 * tmp59
    tmp61 = tmp58 * tmp60
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp51 + tmp62
    tmp64 = 0.0
    tmp65 = tl.where(tmp48, tmp63, tmp64)
    tmp67 = tmp66.to(tl.float32)
    tmp68 = tmp67 * tmp35
    tmp69 = tmp65 * tmp68
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp32, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp65, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp69, rmask)
''')
