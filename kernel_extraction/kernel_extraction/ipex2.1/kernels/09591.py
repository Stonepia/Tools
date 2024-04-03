

# Original file: ./BertForQuestionAnswering__0_backward_207.1/BertForQuestionAnswering__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/ro/crooqjnj7fvgh3dihixeglwpmjhclhertt6kkci37igmkvt5nb2f.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax_backward_data, aten.cat, aten.div, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => full_default_2
triton_per_fused__log_softmax_backward_data_cat_div_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_per_fused__log_softmax_backward_data_cat_div_nll_loss_backward_nll_loss_forward_2', '''
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
    size_hints=[16, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_cat_div_nll_loss_backward_nll_loss_forward_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__log_softmax_backward_data_cat_div_nll_loss_backward_nll_loss_forward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 16
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
    tmp35 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp41 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp42 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask & xmask, other=0.0)
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
    tmp37 = tl.exp(tmp36)
    tmp38 = tmp37 * tmp34
    tmp39 = tmp30 - tmp38
    tmp40 = tmp35 + tmp39
    tmp43 = tl.exp(tmp42)
    tmp44 = tmp43 * tmp20
    tmp45 = tmp16 - tmp44
    tmp46 = tmp41 + tmp45
    tl.store(out_ptr2 + ((2*r1) + (1024*x0)), tmp40, rmask & xmask)
    tl.store(out_ptr3 + ((2*r1) + (1024*x0)), tmp46, rmask & xmask)
''')
