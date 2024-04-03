

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/vi/cvihfuw2plfkxmjaiphupp3uhh67hwfohv4s5mh26gzler6x2mcp.py
# Source Nodes: [eq, masked_fill, softmax, truediv_1], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
# eq => eq
# masked_fill => full_default, where
# softmax => amax, div_2, exp, sub_1, sum_1
# truediv_1 => div_1
triton_red_fused__softmax_div_eq_masked_fill_4 = async_compile.triton('triton_red_fused__softmax_div_eq_masked_fill_4', '''
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_div_eq_masked_fill_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_div_eq_masked_fill_4(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x2 = (xindex // 1536)
    x4 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x0) + (16384*x2)), rmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 == tmp2
        tmp5 = 8.0
        tmp6 = tmp4 / tmp5
        tmp7 = -1000000000.0
        tmp8 = tl.where(tmp3, tmp7, tmp6)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = triton_helpers.maximum(_tmp10, tmp9)
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = triton_helpers.max2(_tmp10, 1)[:, None]
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp12 = tl.load(in_ptr0 + (r3 + (128*x0) + (16384*x2)), rmask, eviction_policy='evict_last')
        tmp16 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp12.to(tl.int64)
        tmp14 = tl.full([1, 1], 0, tl.int64)
        tmp15 = tmp13 == tmp14
        tmp17 = 8.0
        tmp18 = tmp16 / tmp17
        tmp19 = -1000000000.0
        tmp20 = tl.where(tmp15, tmp19, tmp18)
        tmp21 = tmp20 - tmp10
        tmp22 = tl.exp(tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp26 = tl.load(in_ptr0 + (r3 + (128*x0) + (16384*x2)), rmask, eviction_policy='evict_last')
        tmp30 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tmp26.to(tl.int64)
        tmp28 = tl.full([1, 1], 0, tl.int64)
        tmp29 = tmp27 == tmp28
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = -1000000000.0
        tmp34 = tl.where(tmp29, tmp33, tmp32)
        tmp35 = tmp34 - tmp10
        tmp36 = tl.exp(tmp35)
        tmp37 = tmp36 / tmp24
        tl.store(out_ptr2 + (r3 + (128*x4)), tmp37, rmask)
''')
