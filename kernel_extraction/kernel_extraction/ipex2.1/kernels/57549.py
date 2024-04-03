

# Original file: ./DebertaForMaskedLM__0_backward_135.1/DebertaForMaskedLM__0_backward_135.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/dx/cdxrcbza5uiw4j3gdfadh2cvsbsweuh3mayzm7hk2hcocss75vv7.py
# Source Nodes: [float_25, sub_48, trampoline_autograd_apply, truediv_36], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sub, aten.sum]
# float_25 => convert_element_type_207
# sub_48 => sub_97
# trampoline_autograd_apply => full_default_1
# truediv_36 => div_48
triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_10 = async_compile.triton('triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_10', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_masked_fill_mul_neg_pow_sub_sum_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = -tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 / tmp9
    tmp11 = tmp10 / tmp9
    tmp12 = tmp4 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 2.0
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 / tmp18
    tmp20 = 768.0
    tmp21 = tmp19 / tmp20
    tmp22 = tmp8 * tmp17
    tmp23 = tmp21 * tmp22
    tmp24 = -tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp3 / tmp9
    tmp30 = -tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tmp29 + tmp23
    tmp36 = tmp34 + tmp28
    tmp37 = tmp36 / tmp20
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tmp41 = 0.0
    tmp42 = tl.where(tmp40, tmp41, tmp39)
    tmp43 = 1.1111111111111112
    tmp44 = tmp42 * tmp43
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp39, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp44, rmask)
''')
