

# Original file: ./LayoutLMForSequenceClassification__0_backward_135.1/LayoutLMForSequenceClassification__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/ht/chtwi27qc6xii4oxidmhr35j5dtm7mmxjrjkhijlqyxa3y7bsyif.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_22 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_22', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_22', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, xnumel, rnumel):
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
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0)
    tmp22 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (r1 + (768*x0)), rmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp11 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = 768.0
    tmp24 = tmp11 * tmp23
    tmp25 = tmp24 - tmp15
    tmp26 = tmp16 * tmp21
    tmp27 = tmp25 - tmp26
    tmp28 = tmp22 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = 1.1111111111111112
    tmp33 = tmp31 * tmp32
    tmp34 = tmp29 * tmp33
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp28, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp34, rmask)
''')
