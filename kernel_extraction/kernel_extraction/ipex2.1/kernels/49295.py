

# Original file: ./DebertaV2ForQuestionAnswering__0_forward_205.0/DebertaV2ForQuestionAnswering__0_forward_205.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/kq/ckq6corg456exjfanveeqliwh4lhzar6z45ja4rfaatmsqlrtmt6.py
# Source Nodes: [add_48, clamp, clamp_1, cross_entropy, cross_entropy_1, trampoline_autograd_apply, truediv_24], Original ATen: [aten._log_softmax, aten.add, aten.clamp, aten.div, aten.masked_fill, aten.nll_loss_backward, aten.nll_loss_forward]
# add_48 => add_171
# clamp => clamp_max, clamp_min
# clamp_1 => clamp_max_1, clamp_min_1
# cross_entropy => convert_element_type_702, div_48, full_default_170, log, ne, neg, sum_26, sum_27, where_122
# cross_entropy_1 => convert_element_type_704, div_49, log_1, ne_3, neg_1, sum_29, sum_30, where_124
# trampoline_autograd_apply => full_default_1
# truediv_24 => div_50
triton_poi_fused__log_softmax_add_clamp_div_masked_fill_nll_loss_backward_nll_loss_forward_16 = async_compile.triton('triton_poi_fused__log_softmax_add_clamp_div_masked_fill_nll_loss_backward_nll_loss_forward_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*i64', 5: '*fp16', 6: '*fp32', 7: '*fp16', 8: '*fp32', 9: '*i1', 10: '*i1', 11: '*i1', 12: '*i64', 13: '*i1', 14: '*i64', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_add_clamp_div_masked_fill_nll_loss_backward_nll_loss_forward_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]})
@triton.jit
def triton_poi_fused__log_softmax_add_clamp_div_masked_fill_nll_loss_backward_nll_loss_forward_16(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_out_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.load(in_out_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr0 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp13 = tl.load(in_ptr1 + (0))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp22 = tl.load(in_ptr3 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp36 = tl.load(in_ptr5 + (0))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp2 = tl.log(tmp1)
    tmp5 = tl.log(tmp4)
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tl.full([1], 512, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tmp11 != tmp10
    tmp15 = triton_helpers.maximum(tmp14, tmp8)
    tmp16 = triton_helpers.minimum(tmp15, tmp10)
    tmp17 = tmp16 != tmp10
    tmp18 = tl.where(tmp12, tmp11, tmp8)
    tmp19 = tl.where(tmp18 < 0, tmp18 + 512, tmp18)
    # tl.device_assert((0 <= tmp19) & (tmp19 < 512), "index out of bounds: 0 <= tmp19 < 512")
    tmp20 = tl.load(in_ptr2 + (tmp19), None).to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp24 = tmp21 - tmp23
    tmp25 = tmp24 - tmp2
    tmp26 = -tmp25
    tmp27 = 0.0
    tmp28 = tl.where(tmp12, tmp26, tmp27)
    tmp29 = tmp12.to(tl.int64)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 / tmp30
    tmp32 = tl.where(tmp17, tmp16, tmp8)
    tmp33 = tl.where(tmp32 < 0, tmp32 + 512, tmp32)
    # tl.device_assert((0 <= tmp33) & (tmp33 < 512), "index out of bounds: 0 <= tmp33 < 512")
    tmp34 = tl.load(in_ptr4 + (tmp33), None).to(tl.float32)
    tmp35 = tmp34.to(tl.float32)
    tmp38 = tmp35 - tmp37
    tmp39 = tmp38 - tmp5
    tmp40 = -tmp39
    tmp41 = tl.where(tmp17, tmp40, tmp27)
    tmp42 = tmp17.to(tl.int64)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 / tmp43
    tmp45 = tmp31 + tmp44
    tmp46 = 2.0
    tmp47 = tmp45 / tmp46
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None)
    tl.store(in_out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp5, None)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp17, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp17, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp32, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
    tl.store(out_ptr5 + (tl.full([XBLOCK], 0, tl.int32)), tmp18, None)
    tl.store(in_out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp47, None)
''')
