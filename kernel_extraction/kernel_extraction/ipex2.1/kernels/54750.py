

# Original file: ./DebertaV2ForQuestionAnswering__0_backward_207.1/DebertaV2ForQuestionAnswering__0_backward_207.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/ak/cak5ewdgtvgdxvjszn7h2wejlqkb6wnb75w2x3bo6qhmfobpqj5r.py
# Source Nodes: [cross_entropy, cross_entropy_1, trampoline_autograd_apply], Original ATen: [aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.add, aten.cat, aten.div, aten.masked_fill, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => convert_element_type_701, convert_element_type_702, sub_146, sub_147, sum_26
# cross_entropy_1 => convert_element_type_703, convert_element_type_704, sub_148, sub_149, sum_29
# trampoline_autograd_apply => full_default_1
triton_per_fused__log_softmax__log_softmax_backward_data__to_copy_add_cat_div_masked_fill_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_per_fused__log_softmax__log_softmax_backward_data__to_copy_add_cat_div_masked_fill_nll_loss_backward_nll_loss_forward_2', '''
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
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*i1', 6: '*i1', 7: '*fp16', 8: '*fp16', 9: '*fp32', 10: '*fp32', 11: '*fp16', 12: '*fp16', 13: '*fp32', 14: '*fp32', 15: '*fp16', 16: '*fp16', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__log_softmax_backward_data__to_copy_add_cat_div_masked_fill_nll_loss_backward_nll_loss_forward_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__log_softmax__log_softmax_backward_data__to_copy_add_cat_div_masked_fill_nll_loss_backward_nll_loss_forward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr3, out_ptr5, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tl.load(in_ptr2 + (0))
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = tl.load(in_ptr3 + (0))
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp19 = tl.load(in_ptr4 + (r0), rmask, other=0.0)
    tmp20 = tl.load(in_ptr5 + (0))
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp22 = tl.load(in_ptr6 + (0))
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp33 = tl.load(in_ptr7 + (r0), rmask, other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr8 + (r0), rmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr9 + (0))
    tmp37 = tl.broadcast_to(tmp36, [RBLOCK])
    tmp39 = tl.load(in_ptr10 + (0))
    tmp40 = tl.broadcast_to(tmp39, [RBLOCK])
    tmp47 = tl.load(in_ptr11 + (r0), rmask, other=0.0).to(tl.float32)
    tmp48 = tl.load(in_ptr12 + (r0), rmask, other=0.0).to(tl.float32)
    tmp50 = tl.load(in_ptr13 + (0))
    tmp51 = tl.broadcast_to(tmp50, [RBLOCK])
    tmp53 = tl.load(in_ptr14 + (0))
    tmp54 = tl.broadcast_to(tmp53, [RBLOCK])
    tmp5 = 2.0
    tmp6 = tmp4 / tmp5
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp6 / tmp10
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp24 = tmp23.to(tl.int64)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp6 / tmp25
    tmp27 = tl.where(tmp21, tmp26, tmp12)
    tmp28 = tmp19 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp35 = tmp34.to(tl.float32)
    tmp38 = tmp35 - tmp37
    tmp41 = tmp38 - tmp40
    tmp42 = tl.exp(tmp41)
    tmp43 = tmp42 * tmp18
    tmp44 = tmp14 - tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp33 + tmp45
    tmp49 = tmp48.to(tl.float32)
    tmp52 = tmp49 - tmp51
    tmp55 = tmp52 - tmp54
    tmp56 = tl.exp(tmp55)
    tmp57 = tmp56 * tmp32
    tmp58 = tmp28 - tmp57
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp47 + tmp59
    tl.store(out_ptr3 + (tl.broadcast_to(2*r0, [RBLOCK])), tmp46, rmask)
    tl.store(out_ptr5 + (tl.broadcast_to(2*r0, [RBLOCK])), tmp60, rmask)
''')
