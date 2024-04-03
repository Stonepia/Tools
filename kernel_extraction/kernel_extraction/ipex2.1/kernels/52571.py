

# Original file: ./DistilBertForQuestionAnswering__0_backward_99.1/DistilBertForQuestionAnswering__0_backward_99.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/dh/cdhzawvxuwrvvrog7jfebet2r2fyinjofcgmfdgupkpn2ivvxuo7.py
# Source Nodes: [l__mod___distilbert_transformer_layer_5_output_layer_norm], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___distilbert_transformer_layer_5_output_layer_norm => convert_element_type_54
triton_per_fused_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_per_fused_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5', '''
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
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 32768
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask)
    tmp7 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 - tmp16
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = 768.0
    tmp26 = tmp18 / tmp25
    tmp27 = tmp9 * tmp25
    tmp28 = tmp27 - tmp13
    tmp29 = tmp19 * tmp24
    tmp30 = tmp28 - tmp29
    tmp31 = tmp26 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp3
    tmp36 = tmp32 * tmp35
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp32, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp36, rmask)
''')
