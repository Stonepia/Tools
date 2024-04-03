

# Original file: ./hf_Longformer__21_inference_61.1/hf_Longformer__21_inference_61.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/ae/cae3u4ewbfvgzq6b7pylx44gqkfpy56bbxc6pdxekwgpbamfjzmf.py
# Source Nodes: [any_1, gt, lt], Original ATen: [aten.any, aten.gt, aten.lt]
# any_1 => any_1
# gt => gt
# lt => lt
triton_red_fused_any_gt_lt_0 = async_compile.triton('triton_red_fused_any_gt_lt_0', '''
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
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*i1', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_any_gt_lt_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_any_gt_lt_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.int1)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), None, eviction_policy='evict_first')
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 | tmp4
        _tmp5 = tmp6
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp2, None)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp3, None)
    tmp5 = triton_helpers.any(_tmp5.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
''')
