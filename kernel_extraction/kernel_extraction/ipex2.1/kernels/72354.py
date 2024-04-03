

# Original file: ./DebertaForQuestionAnswering__0_backward_135.1/DebertaForQuestionAnswering__0_backward_135.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/6r/c6rzz63ihph2tcdglv2iu3nxosyt67fkz3gcxa2ffaovs2ehtxfg.py
# Source Nodes: [iadd, sub, trampoline_autograd_apply, truediv], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.neg, aten.pow, aten.sub, aten.sum]
# iadd => add
# sub => sub
# trampoline_autograd_apply => full_default_1
# truediv => div
triton_per_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_25 = async_compile.triton('triton_per_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_25', '''
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
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr5'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sub_sum_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask)
    tmp1 = tl.load(in_ptr1 + (r2 + (768*x3)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (r2 + (768*x3)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr5 + (r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x3), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = 1.1111111111111112
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = -tmp9
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 - tmp14
    tmp17 = tmp15 / tmp16
    tmp18 = tmp17 / tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tmp9 / tmp16
    tmp25 = 2.0
    tmp26 = tmp16 * tmp25
    tmp27 = tmp23 / tmp26
    tmp28 = 768.0
    tmp29 = tmp27 / tmp28
    tmp30 = tmp15 * tmp25
    tmp31 = tmp29 * tmp30
    tmp32 = -tmp24
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = -tmp31
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = tl.where(rmask, tmp38, 0)
    tmp41 = triton_helpers.promote_to_tensor(tl.sum(tmp40, 0))
    tmp43 = tl.where(tmp42 < 0, tmp42 + 50265, tmp42)
    tmp44 = tl.full([1], 0, tl.int64)
    tmp45 = tmp42 == tmp44
    tmp46 = tmp24 + tmp31
    tmp47 = tmp36 + tmp41
    tmp48 = tmp47 / tmp28
    tmp49 = tmp46 + tmp48
    tmp50 = tl.where(tmp45, tmp4, tmp49)
    tl.store(out_ptr1 + (r2 + (768*x3)), tmp24, rmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp31, rmask)
    tl.atomic_add(out_ptr5 + (tl.broadcast_to(r2 + (768*tmp43), [RBLOCK])), tmp50, rmask)
    tl.store(out_ptr3 + (x3), tmp36, None)
    tl.store(out_ptr4 + (x3), tmp41, None)
''')
