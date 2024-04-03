

# Original file: ./DebertaForMaskedLM__0_backward_135.1/DebertaForMaskedLM__0_backward_135.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/2d/c2dozvoipf53yihtodpfuk3c2b26y7rkfghyfw44a5ggqlfospu2.py
# Source Nodes: [gelu_12, l__mod___cls_predictions_transform_layer_norm], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# gelu_12 => add_111, convert_element_type_209, erf_12, mul_113
# l__mod___cls_predictions_transform_layer_norm => convert_element_type_211
triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_4 = async_compile.triton('triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_4', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 4096
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
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 - tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tmp4 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 768.0
    tmp21 = tmp13 / tmp20
    tmp22 = tmp4 * tmp20
    tmp23 = tmp22 - tmp8
    tmp24 = tmp14 * tmp19
    tmp25 = tmp23 - tmp24
    tmp26 = tmp21 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = 0.7071067811865476
    tmp30 = tmp28 * tmp29
    tmp31 = libdevice.erf(tmp30)
    tmp32 = 1.0
    tmp33 = tmp31 + tmp32
    tmp34 = 0.5
    tmp35 = tmp33 * tmp34
    tmp36 = tmp28 * tmp28
    tmp37 = -0.5
    tmp38 = tmp36 * tmp37
    tmp39 = tl.exp(tmp38)
    tmp40 = 0.3989422804014327
    tmp41 = tmp39 * tmp40
    tmp42 = tmp28 * tmp41
    tmp43 = tmp35 + tmp42
    tmp44 = tmp26 * tmp43
    tmp45 = tmp44.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp45, rmask)
''')
