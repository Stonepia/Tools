

# Original file: ./PLBartForCausalLM__34_backward_123.11/PLBartForCausalLM__34_backward_123.11_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/5y/c5yyw2nxkmzukk4au2niumqexfqb54jl2ajzzu6zk2fep5c5wgsz.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten._to_copy, aten.native_dropout_backward]

triton_per_fused__softmax_backward_data__to_copy_native_dropout_backward_13 = async_compile.triton('triton_per_fused__softmax_backward_data__to_copy_native_dropout_backward_13', '''
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
    size_hints=[131072, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*i1', 2: '*fp32', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data__to_copy_native_dropout_backward_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_backward_data__to_copy_native_dropout_backward_13(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 98304
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
    tmp2 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0)
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
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp15, rmask)
''')
