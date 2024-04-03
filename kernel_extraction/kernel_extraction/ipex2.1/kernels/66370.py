

# Original file: ./GoogleFnet__0_backward_63.1/GoogleFnet__0_backward_63.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/2g/c2geficak3ycszqtrw2xyj7exqzbrosiclh7iygjzuk26d6zz7di.py
# Source Nodes: [add_50, l__mod___cls_predictions_transform_layer_norm, mul_48, mul_51], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward, aten.pow, aten.tanh_backward]
# add_50 => add_101
# l__mod___cls_predictions_transform_layer_norm => mul_128, sub_25
# mul_48 => mul_124
# mul_51 => mul_127
triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4 = async_compile.triton('triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_tanh_backward_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 * tmp12
    tmp15 = tmp13 - tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tmp2 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = 768.0
    tmp24 = tmp16 / tmp23
    tmp25 = tmp2 * tmp23
    tmp26 = tmp25 - tmp6
    tmp27 = tmp17 * tmp22
    tmp28 = tmp26 - tmp27
    tmp29 = tmp24 * tmp28
    tmp30 = tmp29 * tmp9
    tmp31 = tmp10 * tmp10
    tmp32 = tmp11 - tmp31
    tmp33 = tmp30 * tmp32
    tmp34 = 0.7978845608028654
    tmp35 = tmp33 * tmp34
    tmp36 = 0.044715
    tmp37 = tmp35 * tmp36
    tmp38 = tmp7 * tmp7
    tmp39 = 3.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp37 * tmp40
    tmp42 = tmp35 + tmp41
    tmp43 = tmp29 * tmp12
    tmp44 = tmp43 * tmp8
    tmp45 = tmp42 + tmp44
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp45, rmask)
''')
