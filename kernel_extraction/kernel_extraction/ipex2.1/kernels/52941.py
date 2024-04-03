

# Original file: ./AlbertForQuestionAnswering__0_backward_135.1/AlbertForQuestionAnswering__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/ab/cabdau7gz6opngvemygsmq7d2hybogauqwcsbocpyrz3kvjzbi7q.py
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
    size_hints=[4, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*i64', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_cat_div_nll_loss_backward_nll_loss_forward_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__log_softmax_backward_data_cat_div_nll_loss_backward_nll_loss_forward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp11 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp22 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (0)).to(tl.float32)
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp37 = tl.load(in_ptr7 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr8 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp45 = tl.load(in_ptr9 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp46 = tl.load(in_ptr10 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
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
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp24 = triton_helpers.maximum(tmp23, tmp2)
    tmp25 = triton_helpers.minimum(tmp24, tmp4)
    tmp26 = tmp25 != tmp4
    tmp29 = tmp10 / tmp28
    tmp30 = tl.where(tmp26, tmp29, tmp14)
    tmp31 = tmp22 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask & xmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tl.exp(tmp39)
    tmp41 = tmp40 * tmp36
    tmp42 = tmp32 - tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp37 + tmp43
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tl.exp(tmp47)
    tmp49 = tmp48 * tmp21
    tmp50 = tmp17 - tmp49
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp45 + tmp51
    tl.store(out_ptr2 + ((2*r1) + (1024*x0)), tmp44, rmask & xmask)
    tl.store(out_ptr3 + ((2*r1) + (1024*x0)), tmp52, rmask & xmask)
''')
