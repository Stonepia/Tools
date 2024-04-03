

# Original file: ./DebertaV2ForQuestionAnswering__0_backward_207.1/DebertaV2ForQuestionAnswering__0_backward_207.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/72/c727szb7ju32syejhsoaxnr7iwuh7m5c7wxxjkqdkz4nrbqi4c5s.py
# Source Nodes: [cross_entropy, cross_entropy_1, trampoline_autograd_apply], Original ATen: [aten._log_softmax_backward_data, aten.cat, aten.div, aten.masked_fill, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => convert_element_type_415, sum_26
# cross_entropy_1 => convert_element_type_418, sum_29
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
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*i1', 4: '*fp16', 5: '*i1', 6: '*i1', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_cat_div_masked_fill_nll_loss_backward_nll_loss_forward_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14), equal_to_1=())]}
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
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = tl.load(in_ptr3 + (0))
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp20 = tl.load(in_ptr4 + (r0), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr5 + (0))
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp23 = tl.load(in_ptr6 + (0))
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp35 = tl.load(in_ptr7 + (r0), rmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr8 + (r0), rmask, other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr9 + (r0), rmask, other=0.0).to(tl.float32)
    tmp44 = tl.load(in_ptr10 + (r0), rmask, other=0.0).to(tl.float32)
    tmp5 = 2.0
    tmp6 = tmp4 / tmp5
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp6 / tmp10
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp25 = tmp24.to(tl.int64)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp6 / tmp26
    tmp28 = tl.where(tmp22, tmp27, tmp12)
    tmp29 = tmp20 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tl.exp(tmp37)
    tmp39 = tmp38 * tmp34
    tmp40 = tmp30 - tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp35 + tmp41
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tl.exp(tmp45)
    tmp47 = tmp46 * tmp19
    tmp48 = tmp15 - tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp43 + tmp49
    tl.store(out_ptr2 + (tl.broadcast_to(2*r0, [RBLOCK])), tmp42, rmask)
    tl.store(out_ptr3 + (tl.broadcast_to(2*r0, [RBLOCK])), tmp50, rmask)
''')
