

# Original file: ./MegatronBertForQuestionAnswering__0_backward_207.1/MegatronBertForQuestionAnswering__0_backward_207.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/r5/cr5suudlh33nvl7mafbgngi5wiiwcu2w5cuv6faw7qw2mk7ailku.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_17 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_17', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
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
    tmp2 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp20 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (r1 + (1024*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = tmp9 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp22 = 1024.0
    tmp23 = tmp21 / tmp22
    tmp24 = tmp9 * tmp22
    tmp25 = tmp24 - tmp13
    tmp26 = tmp14 * tmp19
    tmp27 = tmp25 - tmp26
    tmp28 = tmp23 * tmp27
    tmp29 = tmp20 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = 1.1111111111111112
    tmp34 = tmp32 * tmp33
    tmp35 = tmp30 * tmp34
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp29, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp35, rmask)
''')