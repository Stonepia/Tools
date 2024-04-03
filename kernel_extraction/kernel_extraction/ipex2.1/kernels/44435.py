

# Original file: ./BertForMaskedLM__0_backward_135.1/BertForMaskedLM__0_backward_135.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/f2/cf2yw27dcia3tyneb5vxottwez2lojnzlzrqn7ls3o3twz3wdbrt.py
# Source Nodes: [gelu_12, l__self___cls_predictions_transform_layer_norm], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# gelu_12 => add_100, convert_element_type_231, erf_12, mul_162
# l__self___cls_predictions_transform_layer_norm => convert_element_type_233
triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_5', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = 768.0
    tmp20 = tmp12 / tmp19
    tmp21 = tmp3 * tmp19
    tmp22 = tmp21 - tmp7
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp20 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 0.7071067811865476
    tmp29 = tmp27 * tmp28
    tmp30 = libdevice.erf(tmp29)
    tmp31 = 1.0
    tmp32 = tmp30 + tmp31
    tmp33 = 0.5
    tmp34 = tmp32 * tmp33
    tmp35 = tmp27 * tmp27
    tmp36 = -0.5
    tmp37 = tmp35 * tmp36
    tmp38 = tl.exp(tmp37)
    tmp39 = 0.3989422804014327
    tmp40 = tmp38 * tmp39
    tmp41 = tmp27 * tmp40
    tmp42 = tmp34 + tmp41
    tmp43 = tmp25 * tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp44, rmask)
''')
