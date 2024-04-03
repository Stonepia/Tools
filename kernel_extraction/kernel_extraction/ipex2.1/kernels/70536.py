

# Original file: ./hf_T5_generate__68_inference_108.48/hf_T5_generate__68_inference_108.48_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/7w/c7w4l7nxiu54x67segg7jxntnfroq2aikojba6yosiprk6z4hrox.py
# Source Nodes: [softmax_1], Original ATen: [aten._softmax]
# softmax_1 => amax_1, div_3, exp_1, sub_9, sum_2
triton_red_fused__softmax_8 = async_compile.triton('triton_red_fused__softmax_8', '''
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
    size_hints=[8, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_8(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = 1.0
        tmp4 = tmp3 - tmp2
        tmp5 = -3.4028234663852886e+38
        tmp6 = tmp4 * tmp5
        tmp7 = 0.0
        tmp8 = tmp7 + tmp6
        tmp9 = tmp0 + tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = triton_helpers.maximum(_tmp11, tmp10)
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = 1.0
        tmp17 = tmp16 - tmp15
        tmp18 = -3.4028234663852886e+38
        tmp19 = tmp17 * tmp18
        tmp20 = 0.0
        tmp21 = tmp20 + tmp19
        tmp22 = tmp13 + tmp21
        tmp23 = tmp22 - tmp11
        tmp24 = tl.exp(tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp28 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 1.0
        tmp32 = tmp31 - tmp30
        tmp33 = -3.4028234663852886e+38
        tmp34 = tmp32 * tmp33
        tmp35 = 0.0
        tmp36 = tmp35 + tmp34
        tmp37 = tmp28 + tmp36
        tmp38 = tmp37 - tmp11
        tmp39 = tl.exp(tmp38)
        tmp40 = tmp39 / tmp26
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp40, rmask & xmask)
''')
