

# Original file: ./DistilBertForQuestionAnswering__0_backward_99.1/DistilBertForQuestionAnswering__0_backward_99.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/2s/c2sn7tjyf3mljdojcrnd4balkpqe2qgjnidqdleytbr5ncwbzcpk.py
# Source Nodes: [l__mod___distilbert_transformer_layer_4_output_layer_norm], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___distilbert_transformer_layer_4_output_layer_norm => convert_element_type_45
triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_21 = async_compile.triton('triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_21', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp16', 10: '*fp16', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr8 + (r1 + (768*x0)), rmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 - tmp17
    tmp20 = tmp18 * tmp19
    tmp21 = tmp10 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = 768.0
    tmp27 = tmp19 / tmp26
    tmp28 = tmp10 * tmp26
    tmp29 = tmp28 - tmp14
    tmp30 = tmp20 * tmp25
    tmp31 = tmp29 - tmp30
    tmp32 = tmp27 * tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = 1.1111111111111112
    tmp37 = tmp35 * tmp36
    tmp38 = tmp33 * tmp37
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp33, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp38, rmask)
''')
