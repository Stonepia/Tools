

# Original file: ./OPTForCausalLM__43_backward_180.20/OPTForCausalLM__43_backward_180.20_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/6a/c6aozz6u5djdp6j27326wisowqpcq7hrwvz76sonqruopse5aefl.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_7 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_7', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_backward_native_layer_norm_backward_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tmp3 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp16 = 768.0
    tmp17 = tmp3 * tmp16
    tmp18 = tmp17 - tmp7
    tmp19 = tmp8 * tmp13
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tmp22 = tmp14 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = 1.1111111111111112
    tmp27 = tmp25 * tmp26
    tmp28 = tmp23 * tmp27
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp22, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp28, rmask)
''')
