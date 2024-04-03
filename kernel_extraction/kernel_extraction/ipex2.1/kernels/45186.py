

# Original file: ./eca_halonext26ts___60.0/eca_halonext26ts___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/4j/c4jtj6vsnigczkwfjumho7d2bnulduswu76gjf2gimcfkpnsevgi.py
# Source Nodes: [add_6, mul_5, softmax], Original ATen: [aten._softmax, aten.add, aten.mul]
# add_6 => add_50
# mul_5 => mul_90
# softmax => amax, div, exp, sub_22, sum_1
triton_red_fused__softmax_add_mul_16 = async_compile.triton('triton_red_fused__softmax_add_mul_16', '''
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
    size_hints=[262144, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_add_mul_16(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 262144
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp24 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.25
        tmp2 = tmp0 * tmp1
        tmp3 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp4 = tl.full([1, 1], 192, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp7 = tl.full([1, 1], 23, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 8)) + (r2 // 12)) // 24)) + (184*(x0 % 8)) + (1472*((((8*x1) + (x0 % 8)) // 8) % 4096)) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.where(tmp9, tmp10, 0.0)
        tmp12 = tl.where(tmp5, tmp11, 0.0)
        tmp13 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp14 = tmp13 < tmp4
        tmp15 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp16 = tmp15 < tmp7
        tmp17 = tmp16 & tmp14
        tmp18 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 8)) + (r2 % 12)) // 24) % 8)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp17, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.where(tmp17, tmp18, 0.0)
        tmp20 = tl.where(tmp14, tmp19, 0.0)
        tmp21 = tmp12 + tmp20
        tmp22 = tmp2 + tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = triton_helpers.maximum(_tmp24, tmp23)
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp24 = triton_helpers.max2(_tmp24, 1)[:, None]
    _tmp52 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp26 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = 0.25
        tmp28 = tmp26 * tmp27
        tmp29 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp30 = tl.full([1, 1], 192, tl.int64)
        tmp31 = tmp29 < tmp30
        tmp32 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp33 = tl.full([1, 1], 23, tl.int64)
        tmp34 = tmp32 < tmp33
        tmp35 = tmp34 & tmp31
        tmp36 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 8)) + (r2 // 12)) // 24)) + (184*(x0 % 8)) + (1472*((((8*x1) + (x0 % 8)) // 8) % 4096)) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp35, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.where(tmp35, tmp36, 0.0)
        tmp38 = tl.where(tmp31, tmp37, 0.0)
        tmp39 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp40 = tmp39 < tmp30
        tmp41 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp42 = tmp41 < tmp33
        tmp43 = tmp42 & tmp40
        tmp44 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 8)) + (r2 % 12)) // 24) % 8)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp43, eviction_policy='evict_last', other=0.0)
        tmp45 = tl.where(tmp43, tmp44, 0.0)
        tmp46 = tl.where(tmp40, tmp45, 0.0)
        tmp47 = tmp38 + tmp46
        tmp48 = tmp28 + tmp47
        tmp49 = tmp48 - tmp24
        tmp50 = tl.exp(tmp49)
        tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
        tmp53 = _tmp52 + tmp51
        _tmp52 = tl.where(rmask, tmp53, _tmp52)
    tmp52 = tl.sum(_tmp52, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp54 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp55 = 0.25
        tmp56 = tmp54 * tmp55
        tmp57 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp58 = tl.full([1, 1], 192, tl.int64)
        tmp59 = tmp57 < tmp58
        tmp60 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp61 = tl.full([1, 1], 23, tl.int64)
        tmp62 = tmp60 < tmp61
        tmp63 = tmp62 & tmp59
        tmp64 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 8)) + (r2 // 12)) // 24)) + (184*(x0 % 8)) + (1472*((((8*x1) + (x0 % 8)) // 8) % 4096)) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp63, eviction_policy='evict_first', other=0.0)
        tmp65 = tl.where(tmp63, tmp64, 0.0)
        tmp66 = tl.where(tmp59, tmp65, 0.0)
        tmp67 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp68 = tmp67 < tmp58
        tmp69 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp70 = tmp69 < tmp61
        tmp71 = tmp70 & tmp68
        tmp72 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 8)) + (r2 % 12)) // 24) % 8)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp71, eviction_policy='evict_first', other=0.0)
        tmp73 = tl.where(tmp71, tmp72, 0.0)
        tmp74 = tl.where(tmp68, tmp73, 0.0)
        tmp75 = tmp66 + tmp74
        tmp76 = tmp56 + tmp75
        tmp77 = tmp76 - tmp24
        tmp78 = tl.exp(tmp77)
        tmp79 = tmp78 / tmp52
        tl.store(out_ptr2 + (r2 + (144*x3)), tmp79, rmask)
''')
