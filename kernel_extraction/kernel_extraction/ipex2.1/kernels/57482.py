

# Original file: ./DebertaForMaskedLM__0_backward_135.1/DebertaForMaskedLM__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/vb/cvb6hmenvy3vzpp2yj3etnla2w4lihhufnrm2g3icamp5exmcpyg.py
# Source Nodes: [trampoline_autograd_apply_1, truediv_36], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
# trampoline_autograd_apply_1 => full_default_5
# truediv_36 => div_48
triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sum_13 = async_compile.triton('triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sum_13', '''
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
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sum_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sum_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = -tmp3
    tmp7 = tmp5 / tmp6
    tmp8 = tmp7 / tmp6
    tmp9 = tmp4 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = 2.0
    tmp15 = tmp6 * tmp14
    tmp16 = tmp13 / tmp15
    tmp17 = 768.0
    tmp18 = tmp16 / tmp17
    tmp19 = tmp5 * tmp14
    tmp20 = tmp18 * tmp19
    tmp21 = -tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp3 / tmp6
    tmp27 = -tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tmp26 + tmp20
    tmp33 = tmp31 + tmp25
    tmp34 = tmp33 / tmp17
    tmp35 = tmp32 + tmp34
    tmp37 = tmp35.to(tl.float32)
    tmp38 = 0.0
    tmp39 = tl.where(tmp36, tmp38, tmp37)
    tmp40 = 1.1111111111111112
    tmp41 = tmp39 * tmp40
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp35, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp41, rmask)
''')
