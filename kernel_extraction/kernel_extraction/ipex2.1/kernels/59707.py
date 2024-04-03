

# Original file: ./T5ForConditionalGeneration__0_backward_171.1/T5ForConditionalGeneration__0_backward_171.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/hx/chxth6p52d6ckb22lhg3gq74el7c2cibzzteflzbffktxv2zaczt.py
# Source Nodes: [cross_entropy, l__mod___decoder_dropout], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.mul, aten.native_dropout, aten.native_dropout_backward, aten.nll_loss_forward, aten.pow, aten.sum]
# cross_entropy => full_default_7
# l__mod___decoder_dropout => mul_84, mul_85
triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_native_dropout_backward_nll_loss_forward_pow_sum_26 = async_compile.triton('triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_native_dropout_backward_nll_loss_forward_pow_sum_26', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_native_dropout_backward_nll_loss_forward_pow_sum_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_div_embedding_dense_backward_mul_native_dropout_native_dropout_backward_nll_loss_forward_pow_sum_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask)
    tmp9 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask, other=0.0)
    tmp18 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp20 = tmp6 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = -0.5
    tmp23 = tmp17 * tmp22
    tmp24 = tmp19 * tmp19
    tmp25 = tmp24 * tmp19
    tmp26 = tmp23 * tmp25
    tmp27 = 512.0
    tmp28 = tmp26 / tmp27
    tmp29 = 2.0
    tmp30 = tmp12 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp21 + tmp31
    tmp34 = tl.where(tmp33 < 0, tmp33 + 32128, tmp33)
    tmp35 = tl.full([1], -1, tl.int64)
    tmp36 = tmp33 == tmp35
    tmp37 = tmp8 * tmp11
    tmp38 = tmp32 * tmp37
    tmp39 = 0.0
    tmp40 = tl.where(tmp36, tmp39, tmp38)
    tl.atomic_add(out_ptr1 + (tl.broadcast_to(r1 + (512*tmp34), [RBLOCK])), tmp40, rmask)
''')