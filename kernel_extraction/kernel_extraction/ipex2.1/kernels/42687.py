

# Original file: ./MobileBertForMaskedLM__0_backward_354.1/MobileBertForMaskedLM__0_backward_354.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/wa/cwafezsoladju2tjmuorbebucmj27z7w4ddg3jcxcdjnxupeugap.py
# Source Nodes: [l__self___cls_predictions_transform_layer_norm, l__self___cls_predictions_transform_transform_act_fn], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.relu, aten.threshold_backward]
# l__self___cls_predictions_transform_layer_norm => convert_element_type_990
# l__self___cls_predictions_transform_transform_act_fn => relu_96
triton_per_fused_native_layer_norm_native_layer_norm_backward_relu_threshold_backward_9 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_relu_threshold_backward_9', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_relu_threshold_backward_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_relu_threshold_backward_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = triton_helpers.maximum(0, tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 - tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tmp3 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 0.0
    tmp21 = tmp9 <= tmp20
    tmp22 = 512.0
    tmp23 = tmp13 / tmp22
    tmp24 = tmp3 * tmp22
    tmp25 = tmp24 - tmp7
    tmp26 = tmp14 * tmp19
    tmp27 = tmp25 - tmp26
    tmp28 = tmp23 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl.where(tmp21, tmp20, tmp29)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp30, rmask)
''')
