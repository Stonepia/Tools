

# Original file: ./MegatronBertForQuestionAnswering__0_backward_207.1/MegatronBertForQuestionAnswering__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/mv/cmvesebbv6pfa2p3wno5cnstubnhvkwrm6jig2hwamft27evu7y3.py
# Source Nodes: [l__mod___bert_encoder_layer_22_ln], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___bert_encoder_layer_22_ln => convert_element_type_181
triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_19 = async_compile.triton('triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_19', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp36 = tl.load(in_ptr7 + (r1 + (1024*x0)), rmask)
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
    tmp22 = 1024.0
    tmp23 = tmp21 / tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp20 + tmp26
    tmp28 = tmp13 / tmp22
    tmp29 = tmp4 * tmp22
    tmp30 = tmp29 - tmp8
    tmp31 = tmp14 * tmp19
    tmp32 = tmp30 - tmp31
    tmp33 = tmp28 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp27 + tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = 1.1111111111111112
    tmp39 = tmp37 * tmp38
    tmp40 = tmp35 * tmp39
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp35, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp40, rmask)
''')
