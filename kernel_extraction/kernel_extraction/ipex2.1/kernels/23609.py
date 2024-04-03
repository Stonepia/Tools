

# Original file: ./DALLE2_pytorch__41_inference_81.21/DALLE2_pytorch__41_inference_81.21.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/oj/cojxqsmwlduxvisfg72m6lomqrpkhnzfics3np6atnuylpev3reg.py
# Source Nodes: [argmax, cumsum, eq], Original ATen: [aten.argmax, aten.cumsum, aten.eq]
# argmax => argmax
# cumsum => cumsum
# eq => eq
triton_red_fused_argmax_cumsum_eq_13 = async_compile.triton('triton_red_fused_argmax_cumsum_eq_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[1, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*i64', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_argmax_cumsum_eq_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_argmax_cumsum_eq_13(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 77
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp2 = tl.full([XBLOCK, RBLOCK], -9223372036854775808, tl.int64)
    _tmp2_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        _tmp2_next, _tmp2_index_next = triton_helpers.maximum_with_index(
            _tmp2, _tmp2_index, tmp1, rindex
        )
        _tmp2 = tl.where(rmask, _tmp2_next, _tmp2)
        _tmp2_index = tl.where(rmask, _tmp2_index_next, _tmp2_index)
        tmp3 = tl.full([1, 1], 49407, tl.int64)
        tmp4 = tmp0 == tmp3
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp4, rmask)
    _, tmp2_tmp = triton_helpers.max_with_index(_tmp2, _tmp2_index, 1)
    tmp2 = tmp2_tmp[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp2, None)
''')
