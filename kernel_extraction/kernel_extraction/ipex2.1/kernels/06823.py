

# Original file: ./DistilBertForQuestionAnswering__0_forward_97.0/DistilBertForQuestionAnswering__0_forward_97.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/yv/cyvcv6unsbjslioz45kmvwl57lp6i7jjgjmuyeso56dhl3tjl6oh.py
# Source Nodes: [add_13, clamp, clamp_1, cross_entropy, cross_entropy_1, truediv_6], Original ATen: [aten.add, aten.clamp, aten.div, aten.nll_loss_forward]
# add_13 => add_45
# clamp => clamp_max, clamp_min
# clamp_1 => clamp_max_1, clamp_min_1
# cross_entropy => convert_element_type_130, div_12, full_default_7, ne, neg, sum_8, sum_9, where_7
# cross_entropy_1 => convert_element_type_132, div_13, ne_3, neg_1, sum_11, sum_12, where_9
# truediv_6 => div_14
triton_per_fused_add_clamp_div_nll_loss_forward_21 = async_compile.triton('triton_per_fused_add_clamp_div_nll_loss_forward_21', '''
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
    size_hints=[1, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*bf16', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_div_nll_loss_forward_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clamp_div_nll_loss_forward_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp15 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp17 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp26 = tl.load(in_ptr4 + (r0), rmask, other=0.0)
    tmp39 = tl.load(in_ptr6 + (r0), rmask, other=0.0)
    tmp41 = tl.load(in_ptr7 + (r0), rmask, other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tmp4 != tmp3
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.where(tmp5, tmp4, tmp1)
    tmp12 = tl.where(tmp11 < 0, tmp11 + 128, tmp11)
    # tl.device_assert((0 <= tmp12) & (tmp12 < 128), "index out of bounds: 0 <= tmp12 < 128")
    tmp13 = tl.load(in_ptr1 + (tmp12 + (128*r0)), rmask, other=0.0).to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp14 - tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = -tmp18
    tmp20 = 0.0
    tmp21 = tl.where(tmp5, tmp19, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp27 = triton_helpers.maximum(tmp26, tmp1)
    tmp28 = triton_helpers.minimum(tmp27, tmp3)
    tmp29 = tmp28 != tmp3
    tmp30 = tmp29.to(tl.int64)
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = tl.where(tmp29, tmp28, tmp1)
    tmp36 = tl.where(tmp35 < 0, tmp35 + 128, tmp35)
    # tl.device_assert((0 <= tmp36) & (tmp36 < 128), "index out of bounds: 0 <= tmp36 < 128")
    tmp37 = tl.load(in_ptr5 + (tmp36 + (128*r0)), rmask, other=0.0).to(tl.float32)
    tmp38 = tmp37.to(tl.float32)
    tmp40 = tmp38 - tmp39
    tmp42 = tmp40 - tmp41
    tmp43 = -tmp42
    tmp44 = tl.where(tmp29, tmp43, tmp20)
    tmp45 = tl.broadcast_to(tmp44, [RBLOCK])
    tmp47 = tl.where(rmask, tmp45, 0)
    tmp48 = triton_helpers.promote_to_tensor(tl.sum(tmp47, 0))
    tmp49 = tmp34.to(tl.float32)
    tmp50 = tmp10.to(tl.float32)
    tmp51 = tmp48 / tmp49
    tmp52 = tmp25 / tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = 2.0
    tmp55 = tmp53 / tmp54
    tl.store(out_ptr3 + (tl.full([1], 0, tl.int32)), tmp49, None)
    tl.store(out_ptr4 + (tl.full([1], 0, tl.int32)), tmp50, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp55, None)
''')
