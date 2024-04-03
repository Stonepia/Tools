

# Original file: ./DALLE2_pytorch___60.0/DALLE2_pytorch___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/vz/cvzl4glrb5atnlh5acbjlkltwwkudyb27tinnh7p3wj2xuoqptmr.py
# Source Nodes: [argmax, cumsum, eq], Original ATen: [aten.argmax, aten.cumsum, aten.eq]
# argmax => argmax
# cumsum => cumsum
# eq => eq
triton_per_fused_argmax_cumsum_eq_14 = async_compile.triton('triton_per_fused_argmax_cumsum_eq_14', '''
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
    size_hints=[2, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*i64', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_argmax_cumsum_eq_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_argmax_cumsum_eq_14(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 77
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, -9223372036854775808)
    tmp4 = tl.broadcast_to(rindex, tmp3.shape)
    _, tmp2_tmp = triton_helpers.max_with_index(tmp3, tmp4, 1)
    tmp2 = tmp2_tmp[:, None]
    tmp5 = tl.full([1, 1], 49407, tl.int64)
    tmp6 = tmp0 == tmp5
    tl.store(out_ptr1 + (r1 + (77*x0)), tmp6, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')
