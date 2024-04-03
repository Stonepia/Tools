

# Original file: ./GPT2ForSequenceClassification__0_forward_133.0/GPT2ForSequenceClassification__0_forward_133.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/3z/c3zlbug2qi76zodblzxm27jxuezfhdx547lhfzwbwqmihsj3e3i3.py
# Source Nodes: [argmax, eq, int_1, mod, sub], Original ATen: [aten._to_copy, aten.argmax, aten.eq, aten.remainder, aten.sub]
# argmax => argmax
# eq => eq
# int_1 => convert_element_type_170
# mod => remainder
# sub => sub_37
triton_per_fused__to_copy_argmax_eq_remainder_sub_20 = async_compile.triton('triton_per_fused__to_copy_argmax_eq_remainder_sub_20', '''
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
    size_hints=[4, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_argmax_eq_remainder_sub_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_argmax_eq_remainder_sub_20(in_out_ptr0, in_ptr0, xnumel, rnumel):
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2.to(tl.int32)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, -2147483648)
    tmp7 = tl.broadcast_to(rindex, tmp6.shape)
    _, tmp5_tmp = triton_helpers.max_with_index(tmp6, tmp7, 0)
    tmp5 = triton_helpers.promote_to_tensor(tmp5_tmp)
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 - tmp8
    tmp10 = tl.full([1], 1024, tl.int64)
    tmp11 = tmp9 % tmp10
    tmp12 = tmp11 + tmp10
    tmp13 = tl.where(((tmp11 != 0) & ((tmp11 < 0) != (tmp10 < 0))), tmp12, tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp13, xmask)
''')
