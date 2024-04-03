

# Original file: ./DebertaV2ForQuestionAnswering__0_forward_205.0/DebertaV2ForQuestionAnswering__0_forward_205.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/bm/cbmxbh4pwlqlrcwbqfddzgrlptdtgomnbcy3bc7vaezm7zgwlzos.py
# Source Nodes: [add_48, clamp, clamp_1, cross_entropy, cross_entropy_1, trampoline_autograd_apply, truediv_24], Original ATen: [aten.add, aten.clamp, aten.div, aten.masked_fill, aten.nll_loss_backward, aten.nll_loss_forward]
# add_48 => add_171
# clamp => clamp_max, clamp_min
# clamp_1 => clamp_max_1, clamp_min_1
# cross_entropy => convert_element_type_415, div_48, full_default_170, ne, neg, sum_26, sum_27, where_122
# cross_entropy_1 => convert_element_type_418, div_49, ne_3, neg_1, sum_29, sum_30, where_124
# trampoline_autograd_apply => full_default_1
# truediv_24 => div_50
triton_poi_fused_add_clamp_div_masked_fill_nll_loss_backward_nll_loss_forward_9 = async_compile.triton('triton_poi_fused_add_clamp_div_masked_fill_nll_loss_backward_nll_loss_forward_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*i1', 5: '*i1', 6: '*fp16', 7: '*i1', 8: '*i64', 9: '*i1', 10: '*i64', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_masked_fill_nll_loss_backward_nll_loss_forward_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clamp_div_masked_fill_nll_loss_backward_nll_loss_forward_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tl.full([1], 512, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5 != tmp4
    tmp9 = triton_helpers.maximum(tmp8, tmp2)
    tmp10 = triton_helpers.minimum(tmp9, tmp4)
    tmp11 = tmp10 != tmp4
    tmp12 = tl.where(tmp6, tmp5, tmp2)
    tmp13 = tl.where(tmp12 < 0, tmp12 + 512, tmp12)
    # tl.device_assert((0 <= tmp13) & (tmp13 < 512), "index out of bounds: 0 <= tmp13 < 512")
    tmp14 = tl.load(in_ptr2 + (tmp13), None).to(tl.float32)
    tmp15 = -tmp14
    tmp16 = 0.0
    tmp17 = tl.where(tmp6, tmp15, tmp16)
    tmp18 = tmp6.to(tl.int64)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tl.where(tmp11, tmp10, tmp2)
    tmp22 = tl.where(tmp21 < 0, tmp21 + 512, tmp21)
    # tl.device_assert((0 <= tmp22) & (tmp22 < 512), "index out of bounds: 0 <= tmp22 < 512")
    tmp23 = tl.load(in_ptr3 + (tmp22), None).to(tl.float32)
    tmp24 = -tmp23
    tmp25 = tl.where(tmp11, tmp24, tmp16)
    tmp26 = tmp11.to(tl.int64)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp20 + tmp28
    tmp30 = 2.0
    tmp31 = tmp29 / tmp30
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp31, None)
    tl.store(out_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp21, None)
    tl.store(out_ptr5 + (tl.full([XBLOCK], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr6 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
''')
