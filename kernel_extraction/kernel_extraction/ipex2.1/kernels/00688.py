

# Original file: ./YituTechConvBert__0_backward_171.1/YituTechConvBert__0_backward_171.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/5h/c5huuapf356ay7eidkbgyha4hlazrmqt6fgt7wkxerrgwjfp6r3f.py
# Source Nodes: [l__mod___convbert_encoder_layer_10_output_layer_norm], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___convbert_encoder_layer_10_output_layer_norm => convert_element_type_133
triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_31 = async_compile.triton('triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_31', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp32', 9: '*fp32', 10: '*i1', 11: '*fp32', 12: '*fp16', 13: '*fp16', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr3, out_ptr4, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + (512*r2) + (393216*x1)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr5 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr7 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr8 + (x3), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x3), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (r2 + (768*x3)), rmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp20 - tmp21
    tmp24 = tmp22 * tmp23
    tmp25 = tmp14 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = 768.0
    tmp31 = tmp23 / tmp30
    tmp32 = tmp14 * tmp30
    tmp33 = tmp32 - tmp18
    tmp34 = tmp24 * tmp29
    tmp35 = tmp33 - tmp34
    tmp36 = tmp31 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = 1.1111111111111112
    tmp41 = tmp39 * tmp40
    tmp42 = tmp37 * tmp41
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp11, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp37, rmask)
    tl.store(out_ptr4 + (r2 + (768*x3)), tmp42, rmask)
''')
