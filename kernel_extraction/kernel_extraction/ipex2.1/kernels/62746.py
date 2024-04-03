

# Original file: ./DebertaForQuestionAnswering__0_forward_133.0/DebertaForQuestionAnswering__0_forward_133.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/75/c75a3airk3kp6ccg6sxdfr2kaa76vviszxqzx3zr6p43kqelyagk.py
# Source Nodes: [contiguous_13, cross_entropy_1], Original ATen: [aten._log_softmax, aten.clone]
# contiguous_13 => clone_49
# cross_entropy_1 => amax_13, convert_element_type_212, convert_element_type_213, exp_13, log_1, sub_101, sub_102, sum_16
triton_per_fused__log_softmax_clone_10 = async_compile.triton('triton_per_fused__log_softmax_clone_10', '''
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
    size_hints=[16, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_clone_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__log_softmax_clone_10(in_ptr0, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (1 + (2*r1) + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, float("-inf"))
    tmp5 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp4, 0))
    tmp6 = tmp1 - tmp5
    tmp7 = tl.exp(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.log(tmp11)
    tmp13 = tmp6 - tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp0, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp14, rmask & xmask)
''')
