

# Original file: ./MobileBertForMaskedLM__0_backward_354.1/MobileBertForMaskedLM__0_backward_354.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/vo/cvohzourunotq5st75y64bntazca7zlcegj6xuawevzw4zsjqkgh.py
# Source Nodes: [cross_entropy, l__mod___cls_predictions_transform_layer_norm, l__mod___cls_predictions_transform_transform_act_fn], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.nll_loss_forward, aten.relu, aten.threshold_backward]
# cross_entropy => full_default_3
# l__mod___cls_predictions_transform_layer_norm => mul_242, sub_25
# l__mod___cls_predictions_transform_transform_act_fn => relu_96
triton_per_fused_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_threshold_backward_6 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_threshold_backward_6', '''
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
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_threshold_backward_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_threshold_backward_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 16384
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp10 = tmp8 - tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp2 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = 0.0
    tmp19 = tmp8 <= tmp18
    tmp20 = 512.0
    tmp21 = tmp11 / tmp20
    tmp22 = tmp2 * tmp20
    tmp23 = tmp22 - tmp6
    tmp24 = tmp12 * tmp17
    tmp25 = tmp23 - tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tl.where(tmp19, tmp18, tmp26)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask)
''')
