

# Original file: ./DebertaForMaskedLM__0_backward_135.1/DebertaForMaskedLM__0_backward_135.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/nw/cnwoxwez4fz2iqno3xpeq75dtt7qciclsajlk4vw4qrxsuspcmcw.py
# Source Nodes: [trampoline_autograd_apply, truediv_36], Original ATen: [aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sum]
# trampoline_autograd_apply => full_default_1
# truediv_36 => div_48
triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_9 = async_compile.triton('triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_9', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_div_masked_fill_mul_neg_pow_sum_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask)
    tmp2 = tmp0 * tmp1
    tmp3 = -tmp2
    tmp6 = tmp4 / tmp5
    tmp7 = tmp6 / tmp5
    tmp8 = tmp3 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 2.0
    tmp14 = tmp5 * tmp13
    tmp15 = tmp12 / tmp14
    tmp16 = 768.0
    tmp17 = tmp15 / tmp16
    tmp18 = tmp4 * tmp13
    tmp19 = tmp17 * tmp18
    tmp20 = -tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp2 / tmp5
    tmp26 = -tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp31 = tmp25 + tmp19
    tmp32 = tmp30 + tmp24
    tmp33 = tmp32 / tmp16
    tmp34 = tmp31 + tmp33
    tmp36 = 0.0
    tmp37 = tl.where(tmp35, tmp36, tmp34)
    tmp38 = 1.1111111111111112
    tmp39 = tmp37 * tmp38
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp34, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp39, rmask)
''')
