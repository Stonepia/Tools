

# Original file: ./sam___60.0/sam___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/7q/c7qxuo6h46cofnv6nxzq3lsbcozl5jth5cq2mjjcp4rl6ysiwhwq.py
# Source Nodes: [softmax_7], Original ATen: [aten._softmax]
# softmax_7 => amax_7, div_8, exp_7, sub_39, sum_8
triton_red_fused__softmax_20 = async_compile.triton('triton_red_fused__softmax_20', '''
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
    size_hints=[65536, 4096],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_20(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    _tmp6 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (4096*x3)), None, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + ((64*(x0 % 64)) + (4096*x1) + (65536*(x0 // 64)) + (r2 // 64)), None, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + ((64*(x0 // 64)) + (4096*x1) + (65536*(x0 % 64)) + (r2 % 64)), None, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = triton_helpers.maximum(_tmp6, tmp5)
        _tmp6 = tmp7
    tmp6 = triton_helpers.max2(_tmp6, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(in_ptr0 + (r2 + (4096*x3)), None, eviction_policy='evict_last')
        tmp9 = tl.load(in_ptr1 + ((64*(x0 % 64)) + (4096*x1) + (65536*(x0 // 64)) + (r2 // 64)), None, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr2 + ((64*(x0 // 64)) + (4096*x1) + (65536*(x0 % 64)) + (r2 % 64)), None, eviction_policy='evict_last')
        tmp10 = tmp8 + tmp9
        tmp12 = tmp10 + tmp11
        tmp13 = tmp12 - tmp6
        tmp14 = tl.exp(tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tmp17
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp18 = tl.load(in_ptr0 + (r2 + (4096*x3)), None, eviction_policy='evict_first')
        tmp19 = tl.load(in_ptr1 + ((64*(x0 % 64)) + (4096*x1) + (65536*(x0 // 64)) + (r2 // 64)), None, eviction_policy='evict_first')
        tmp21 = tl.load(in_ptr2 + ((64*(x0 // 64)) + (4096*x1) + (65536*(x0 % 64)) + (r2 % 64)), None, eviction_policy='evict_first')
        tmp20 = tmp18 + tmp19
        tmp22 = tmp20 + tmp21
        tmp23 = tmp22 - tmp6
        tmp24 = tl.exp(tmp23)
        tmp25 = tmp24 / tmp16
        tl.store(out_ptr2 + (r2 + (4096*x3)), tmp25, None)
''')
