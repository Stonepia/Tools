

# Original file: ./sebotnet33ts_256___60.0/sebotnet33ts_256___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/f3/cf32wqht3x7uqo35y337hofr3efhyyomfooskhdwvlu6hqxt64pd.py
# Source Nodes: [add_5, mul_4, softmax], Original ATen: [aten._softmax, aten.add, aten.mul]
# add_5 => add_41
# mul_4 => mul_74
# softmax => amax, div, exp, sub_18, sum_1
triton_red_fused__softmax_add_mul_17 = async_compile.triton('triton_red_fused__softmax_add_mul_17', '''
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
    size_hints=[262144, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_add_mul_17(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 262144
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp24 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.1767766952966369
        tmp2 = tmp0 * tmp1
        tmp3 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp4 = tl.full([1, 1], 2048, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp7 = tl.full([1, 1], 63, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*((((32*x1) + (x0 % 32)) // 32) % 256)) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.where(tmp9, tmp10, 0.0)
        tmp12 = tl.where(tmp5, tmp11, 0.0)
        tmp13 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp14 = tmp13 < tmp4
        tmp15 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp16 = tmp15 < tmp7
        tmp17 = tmp16 & tmp14
        tmp18 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp17, eviction_policy='evict_last', other=0.0)
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
        tmp26 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = 0.1767766952966369
        tmp28 = tmp26 * tmp27
        tmp29 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp30 = tl.full([1, 1], 2048, tl.int64)
        tmp31 = tmp29 < tmp30
        tmp32 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp33 = tl.full([1, 1], 63, tl.int64)
        tmp34 = tmp32 < tmp33
        tmp35 = tmp34 & tmp31
        tmp36 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*((((32*x1) + (x0 % 32)) // 32) % 256)) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp35, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.where(tmp35, tmp36, 0.0)
        tmp38 = tl.where(tmp31, tmp37, 0.0)
        tmp39 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp40 = tmp39 < tmp30
        tmp41 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp42 = tmp41 < tmp33
        tmp43 = tmp42 & tmp40
        tmp44 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp43, eviction_policy='evict_last', other=0.0)
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
        tmp54 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp55 = 0.1767766952966369
        tmp56 = tmp54 * tmp55
        tmp57 = 31 + (63*(x0 // 32)) + (r2 // 32)
        tmp58 = tl.full([1, 1], 2048, tl.int64)
        tmp59 = tmp57 < tmp58
        tmp60 = (31 + (63*(x0 // 32)) + (r2 // 32)) % 64
        tmp61 = tl.full([1, 1], 63, tl.int64)
        tmp62 = tmp60 < tmp61
        tmp63 = tmp62 & tmp59
        tmp64 = tl.load(in_ptr1 + ((63*((31 + (63*(x0 // 32)) + (r2 // 32)) // 64)) + (2016*(x0 % 32)) + (64512*((((32*x1) + (x0 % 32)) // 32) % 256)) + ((31 + (63*(x0 // 32)) + (r2 // 32)) % 64)), rmask & tmp63, eviction_policy='evict_first', other=0.0)
        tmp65 = tl.where(tmp63, tmp64, 0.0)
        tmp66 = tl.where(tmp59, tmp65, 0.0)
        tmp67 = 31 + (63*(x0 % 32)) + (r2 % 32)
        tmp68 = tmp67 < tmp58
        tmp69 = (31 + (63*(x0 % 32)) + (r2 % 32)) % 64
        tmp70 = tmp69 < tmp61
        tmp71 = tmp70 & tmp68
        tmp72 = tl.load(in_ptr2 + ((63*(((31 + (63*(x0 % 32)) + (r2 % 32)) // 64) % 32)) + (2016*(x0 // 32)) + (64512*x1) + ((31 + (63*(x0 % 32)) + (r2 % 32)) % 64)), rmask & tmp71, eviction_policy='evict_first', other=0.0)
        tmp73 = tl.where(tmp71, tmp72, 0.0)
        tmp74 = tl.where(tmp68, tmp73, 0.0)
        tmp75 = tmp66 + tmp74
        tmp76 = tmp56 + tmp75
        tmp77 = tmp76 - tmp24
        tmp78 = tl.exp(tmp77)
        tmp79 = tmp78 / tmp52
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp79, rmask)
''')
