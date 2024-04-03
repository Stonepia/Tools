

# Original file: ./hf_BigBird___60.0/hf_BigBird___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/mf/cmf27icshokixox53zpnd5yjolrcevvj2xnoq7n34hydn6szadu4.py
# Source Nodes: [iadd_1, softmax], Original ATen: [aten._softmax, aten.add]
# iadd_1 => mul_6
# softmax => amax, convert_element_type_6, convert_element_type_7, div_2, exp, sub_2, sum_1
triton_red_fused__softmax_add_3 = async_compile.triton('triton_red_fused__softmax_add_3', '''
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
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_add_3(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = 0.125
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(xmask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr0 + (r1 + (4096*x0)), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = 0.125
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 - tmp5
        tmp12 = tl.exp(tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr0 + (r1 + (4096*x0)), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = 0.125
        tmp18 = tmp16 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 - tmp5
        tmp21 = tl.exp(tmp20)
        tmp22 = tmp21 / tmp14
        tmp23 = tmp22.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp23, xmask)
''')
