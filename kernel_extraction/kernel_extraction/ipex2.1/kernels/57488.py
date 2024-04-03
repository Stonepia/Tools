

# Original file: ./DebertaForMaskedLM__0_backward_135.1/DebertaForMaskedLM__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/4o/c4opfcrc2duh3phjzlcnhje3vx5mt4bsre2lqnqnhfvxdzmnbvyd.py
# Source Nodes: [trampoline_autograd_apply_1, truediv_35], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
# trampoline_autograd_apply_1 => full_default_5
# truediv_35 => div_47
triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sum_19 = async_compile.triton('triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sum_19', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sum_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sum_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = -tmp5
    tmp9 = tmp7 / tmp8
    tmp10 = tmp9 / tmp8
    tmp11 = tmp6 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tmp5 / tmp8
    tmp17 = -tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = 2.0
    tmp23 = tmp8 * tmp22
    tmp24 = tmp15 / tmp23
    tmp25 = 768.0
    tmp26 = tmp24 / tmp25
    tmp27 = tmp7 * tmp22
    tmp28 = tmp26 * tmp27
    tmp29 = -tmp28
    tmp30 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp32 = tl.where(rmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tmp16 + tmp28
    tmp35 = tmp21 + tmp33
    tmp36 = tmp35 / tmp25
    tmp37 = tmp34 + tmp36
    tmp39 = tmp37.to(tl.float32)
    tmp40 = 0.0
    tmp41 = tl.where(tmp38, tmp40, tmp39)
    tmp42 = 1.1111111111111112
    tmp43 = tmp41 * tmp42
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp37, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp43, rmask)
''')
