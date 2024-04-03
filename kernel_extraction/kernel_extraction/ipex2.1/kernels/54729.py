

# Original file: ./DebertaV2ForQuestionAnswering__0_backward_207.1/DebertaV2ForQuestionAnswering__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/7w/c7wdxzfrjma3pecdt5cchzmsz5he6krjcpw5jqp354ismhdx5am2.py
# Source Nodes: [cross_entropy, cross_entropy_1, trampoline_autograd_apply], Original ATen: [aten._log_softmax_backward_data, aten.cat, aten.div, aten.masked_fill, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => convert_element_type_170, sum_26
# cross_entropy_1 => convert_element_type_171, sum_29
# trampoline_autograd_apply => full_default_1
triton_per_fused__log_softmax_backward_data_cat_div_masked_fill_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_per_fused__log_softmax_backward_data_cat_div_masked_fill_nll_loss_backward_nll_loss_forward_2', '''
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
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*i1', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_cat_div_masked_fill_nll_loss_backward_nll_loss_forward_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__log_softmax_backward_data_cat_div_masked_fill_nll_loss_backward_nll_loss_forward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp33 = tl.load(in_ptr7 + (r0), rmask, other=0.0)
    tmp34 = tl.load(in_ptr8 + (r0), rmask, other=0.0)
    tmp39 = tl.load(in_ptr9 + (r0), rmask, other=0.0)
    tmp40 = tl.load(in_ptr10 + (r0), rmask, other=0.0)
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
    tmp35 = tl.exp(tmp34)
    tmp36 = tmp35 * tmp32
    tmp37 = tmp28 - tmp36
    tmp38 = tmp33 + tmp37
    tmp41 = tl.exp(tmp40)
    tmp42 = tmp41 * tmp18
    tmp43 = tmp14 - tmp42
    tmp44 = tmp39 + tmp43
    tl.store(out_ptr2 + (tl.broadcast_to(2*r0, [RBLOCK])), tmp38, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(2*r0, [RBLOCK])), tmp44, rmask)
''')
