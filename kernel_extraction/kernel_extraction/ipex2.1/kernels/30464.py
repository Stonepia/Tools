

# Original file: ./XLNetLMHeadModel__0_backward_567.1/XLNetLMHeadModel__0_backward_567.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/ku/ckujfudn2vwtz4flhwsc4qsj2xhf7nf6yd4oe6lgt7vrfea4qkix.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.index_add, aten.mul, aten.native_dropout_backward, aten.new_zeros]

triton_per_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14 = async_compile.triton('triton_per_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14', '''
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
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_backward_data_index_add_mul_native_dropout_backward_new_zeros_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask)
    tmp6 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tl.where(tmp12 < 0, tmp12 + 1023, tmp12)
    # tl.device_assert(((0 <= tmp13) & (tmp13 < 1023)) | ~rmask, "index out of bounds: 0 <= tmp13 < 1023")
    tmp14 = tmp6 * tmp11
    tmp15 = tmp7 - tmp14
    tmp16 = 0.125
    tmp17 = tmp15 * tmp16
    tl.atomic_add(out_ptr1 + (tl.broadcast_to(tmp13 + (1023*x0), [RBLOCK])), tmp17, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp17, rmask)
''')
