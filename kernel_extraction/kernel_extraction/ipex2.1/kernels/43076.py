

# Original file: ./DALLE2_pytorch__25_inference_65.5/DALLE2_pytorch__25_inference_65.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/gy/cgytip2nvd43wirnjsyc52wytsnj7kmnltosq2lm4dtpppi42du5.py
# Source Nodes: [add_3, masked_fill, ones, softmax, triu], Original ATen: [aten._softmax, aten.add, aten.masked_fill, aten.ones, aten.triu]
# add_3 => add_3
# masked_fill => full_default_1, where
# ones => full_default
# softmax => amax, div_2, exp, sub_2, sum_3
# triu => ge, logical_and, sub_1
triton_red_fused__softmax_add_masked_fill_ones_triu_13 = async_compile.triton('triton_red_fused__softmax_add_masked_fill_ones_triu_13', '''
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
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_masked_fill_ones_triu_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_add_masked_fill_ones_triu_13(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 261
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 260
    x5 = xindex
    x1 = (xindex // 260) % 8
    _tmp11 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp5 = tl.load(in_ptr0 + (r3 + (261*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (x1 + (8*r3) + (2088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = r3 + ((-1)*x0)
        tmp1 = tl.full([1, 1], 2, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], True, tl.int1)
        tmp4 = tmp2 & tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = -3.4028234663852886e+38
        tmp9 = tl.where(tmp4, tmp8, tmp7)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = triton_helpers.maximum(_tmp11, tmp10)
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp18 = tl.load(in_ptr0 + (r3 + (261*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr1 + (x1 + (8*r3) + (2088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = r3 + ((-1)*x0)
        tmp14 = tl.full([1, 1], 2, tl.int64)
        tmp15 = tmp13 >= tmp14
        tmp16 = tl.full([1, 1], True, tl.int1)
        tmp17 = tmp15 & tmp16
        tmp20 = tmp18 + tmp19
        tmp21 = -3.4028234663852886e+38
        tmp22 = tl.where(tmp17, tmp21, tmp20)
        tmp23 = tmp22 - tmp11
        tmp24 = tl.exp(tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp33 = tl.load(in_ptr0 + (r3 + (261*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp34 = tl.load(in_ptr1 + (x1 + (8*r3) + (2088*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = r3 + ((-1)*x0)
        tmp29 = tl.full([1, 1], 2, tl.int64)
        tmp30 = tmp28 >= tmp29
        tmp31 = tl.full([1, 1], True, tl.int1)
        tmp32 = tmp30 & tmp31
        tmp35 = tmp33 + tmp34
        tmp36 = -3.4028234663852886e+38
        tmp37 = tl.where(tmp32, tmp36, tmp35)
        tmp38 = tmp37 - tmp11
        tmp39 = tl.exp(tmp38)
        tmp40 = tmp39 / tmp26
        tl.store(out_ptr2 + (r3 + (261*x5)), tmp40, rmask & xmask)
''')
