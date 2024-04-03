

# Original file: ./ElectraForCausalLM__0_backward_171.1/ElectraForCausalLM__0_backward_171.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/rm/crmnxagyepfhqypdegf5wfprmqj2vkq2i7y23z72bjkutflfhsjq.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten._to_copy, aten.div, aten.native_dropout_backward]

triton_per_fused__softmax_backward_data__to_copy_div_native_dropout_backward_25 = async_compile.triton('triton_per_fused__softmax_backward_data__to_copy_div_native_dropout_backward_25', '''
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
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i1', 2: '*fp32', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data__to_copy_div_native_dropout_backward_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_backward_data__to_copy_div_native_dropout_backward_25(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 65536
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
    tmp2 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tmp7 * tmp12
    tmp14 = tmp8 - tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = 8.0
    tmp17 = tmp15 / tmp16
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp17, rmask)
''')
