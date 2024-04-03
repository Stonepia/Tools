

# Original file: ./AlbertForQuestionAnswering__0_backward_135.1/AlbertForQuestionAnswering__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/uf/cuf2bhz432mmtcl4m7u662q7c6kmshcxyxnwnxra3fmweskdycn7.py
# Source Nodes: [cross_entropy, cross_entropy_1], Original ATen: [aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.add, aten.cat, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => convert_element_type_233, full_default_2, sub_38, sub_39
# cross_entropy_1 => convert_element_type_235, sub_40, sub_41
triton_per_fused__log_softmax__log_softmax_backward_data__to_copy_add_cat_div_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_per_fused__log_softmax__log_softmax_backward_data__to_copy_add_cat_div_nll_loss_backward_nll_loss_forward_2', '''
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
    size_hints=[4, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*bf16', 8: '*bf16', 9: '*fp32', 10: '*fp32', 11: '*bf16', 12: '*bf16', 13: '*fp32', 14: '*fp32', 15: '*bf16', 16: '*bf16', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__log_softmax_backward_data__to_copy_add_cat_div_nll_loss_backward_nll_loss_forward_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__log_softmax__log_softmax_backward_data__to_copy_add_cat_div_nll_loss_backward_nll_loss_forward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr3, out_ptr5, xnumel, rnumel):
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp11 = tl.load(in_ptr3 + (0))
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp21 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp22 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (0))
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp35 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr11 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp48 = tl.load(in_ptr12 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp50 = tl.load(in_ptr13 + (x0), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr14 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tl.full([1], 512, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5 != tmp4
    tmp9 = 2.0
    tmp10 = tmp8 / tmp9
    tmp13 = tmp10 / tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tmp0 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp23 = triton_helpers.maximum(tmp22, tmp2)
    tmp24 = triton_helpers.minimum(tmp23, tmp4)
    tmp25 = tmp24 != tmp4
    tmp28 = tmp10 / tmp27
    tmp29 = tl.where(tmp25, tmp28, tmp14)
    tmp30 = tmp21 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp37 = tmp36.to(tl.float32)
    tmp39 = tmp37 - tmp38
    tmp41 = tmp39 - tmp40
    tmp42 = tl.exp(tmp41)
    tmp43 = tmp42 * tmp20
    tmp44 = tmp16 - tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp35 + tmp45
    tmp49 = tmp48.to(tl.float32)
    tmp51 = tmp49 - tmp50
    tmp53 = tmp51 - tmp52
    tmp54 = tl.exp(tmp53)
    tmp55 = tmp54 * tmp34
    tmp56 = tmp30 - tmp55
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp47 + tmp57
    tl.store(out_ptr3 + ((2*r1) + (1024*x0)), tmp46, rmask & xmask)
    tl.store(out_ptr5 + ((2*r1) + (1024*x0)), tmp58, rmask & xmask)
''')
