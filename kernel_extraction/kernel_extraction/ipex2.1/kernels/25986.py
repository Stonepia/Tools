

# Original file: ./hf_T5___60.0/hf_T5___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/hr/chr72tkkvnu463ysxyyff63lmycnu3gmoxiijj2ofkmxhytsuuqg.py
# Source Nodes: [softmax_6], Original ATen: [aten._softmax]
# softmax_6 => amax_6, div_10, exp_6, sub_11, sum_7
triton_red_fused__softmax_2 = async_compile.triton('triton_red_fused__softmax_2', '''
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
    size_hints=[16384, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_2(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp31 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = (-1)*(tl.minimum(0, r2 + ((-1)*x0), tl.PropagateNan.NONE))
        tmp2 = tl.full([1, 1], 16, tl.int64)
        tmp3 = tmp1 < tmp2
        tmp4 = tmp1.to(tl.float32)
        tmp5 = 16.0
        tmp6 = tmp4 / tmp5
        tmp7 = tl.log(tmp6)
        tmp8 = 2.0794415416798357
        tmp9 = tmp7 / tmp8
        tmp10 = tmp9 * tmp5
        tmp11 = tmp10.to(tl.int64)
        tmp12 = tmp11 + tmp2
        tmp13 = tl.full([1, 1], 31, tl.int64)
        tmp14 = triton_helpers.minimum(tmp12, tmp13)
        tmp15 = tl.where(tmp3, tmp1, tmp14)
        tmp16 = tl.full([1, 1], 0, tl.int64)
        tmp17 = tmp15 + tmp16
        tmp18 = tl.where(tmp17 < 0, tmp17 + 32, tmp17)
        # tl.device_assert((0 <= tmp18) & (tmp18 < 32), "index out of bounds: 0 <= tmp18 < 32")
        tmp19 = tl.load(in_ptr1 + (x1 + (8*tmp18)), None, eviction_policy='evict_last')
        tmp20 = r2
        tmp21 = x0
        tmp22 = tmp20 <= tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = 1.0
        tmp25 = tmp24 - tmp23
        tmp26 = -3.4028234663852886e+38
        tmp27 = tmp25 * tmp26
        tmp28 = tmp19 + tmp27
        tmp29 = tmp0 + tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = triton_helpers.maximum(_tmp31, tmp30)
        _tmp31 = tl.where(rmask, tmp32, _tmp31)
    tmp31 = triton_helpers.max2(_tmp31, 1)[:, None]
    _tmp66 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp33 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = (-1)*(tl.minimum(0, r2 + ((-1)*x0), tl.PropagateNan.NONE))
        tmp35 = tl.full([1, 1], 16, tl.int64)
        tmp36 = tmp34 < tmp35
        tmp37 = tmp34.to(tl.float32)
        tmp38 = 16.0
        tmp39 = tmp37 / tmp38
        tmp40 = tl.log(tmp39)
        tmp41 = 2.0794415416798357
        tmp42 = tmp40 / tmp41
        tmp43 = tmp42 * tmp38
        tmp44 = tmp43.to(tl.int64)
        tmp45 = tmp44 + tmp35
        tmp46 = tl.full([1, 1], 31, tl.int64)
        tmp47 = triton_helpers.minimum(tmp45, tmp46)
        tmp48 = tl.where(tmp36, tmp34, tmp47)
        tmp49 = tl.full([1, 1], 0, tl.int64)
        tmp50 = tmp48 + tmp49
        tmp51 = tl.where(tmp50 < 0, tmp50 + 32, tmp50)
        # tl.device_assert((0 <= tmp51) & (tmp51 < 32), "index out of bounds: 0 <= tmp51 < 32")
        tmp52 = tl.load(in_ptr1 + (x1 + (8*tmp51)), None, eviction_policy='evict_last')
        tmp53 = r2
        tmp54 = x0
        tmp55 = tmp53 <= tmp54
        tmp56 = tmp55.to(tl.float32)
        tmp57 = 1.0
        tmp58 = tmp57 - tmp56
        tmp59 = -3.4028234663852886e+38
        tmp60 = tmp58 * tmp59
        tmp61 = tmp52 + tmp60
        tmp62 = tmp33 + tmp61
        tmp63 = tmp62 - tmp31
        tmp64 = tl.exp(tmp63)
        tmp65 = tl.broadcast_to(tmp64, [XBLOCK, RBLOCK])
        tmp67 = _tmp66 + tmp65
        _tmp66 = tl.where(rmask, tmp67, _tmp66)
    tmp66 = tl.sum(_tmp66, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp68 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp69 = (-1)*(tl.minimum(0, r2 + ((-1)*x0), tl.PropagateNan.NONE))
        tmp70 = tl.full([1, 1], 16, tl.int64)
        tmp71 = tmp69 < tmp70
        tmp72 = tmp69.to(tl.float32)
        tmp73 = 16.0
        tmp74 = tmp72 / tmp73
        tmp75 = tl.log(tmp74)
        tmp76 = 2.0794415416798357
        tmp77 = tmp75 / tmp76
        tmp78 = tmp77 * tmp73
        tmp79 = tmp78.to(tl.int64)
        tmp80 = tmp79 + tmp70
        tmp81 = tl.full([1, 1], 31, tl.int64)
        tmp82 = triton_helpers.minimum(tmp80, tmp81)
        tmp83 = tl.where(tmp71, tmp69, tmp82)
        tmp84 = tl.full([1, 1], 0, tl.int64)
        tmp85 = tmp83 + tmp84
        tmp86 = tl.where(tmp85 < 0, tmp85 + 32, tmp85)
        # tl.device_assert((0 <= tmp86) & (tmp86 < 32), "index out of bounds: 0 <= tmp86 < 32")
        tmp87 = tl.load(in_ptr1 + (x1 + (8*tmp86)), None, eviction_policy='evict_first')
        tmp88 = r2
        tmp89 = x0
        tmp90 = tmp88 <= tmp89
        tmp91 = tmp90.to(tl.float32)
        tmp92 = 1.0
        tmp93 = tmp92 - tmp91
        tmp94 = -3.4028234663852886e+38
        tmp95 = tmp93 * tmp94
        tmp96 = tmp87 + tmp95
        tmp97 = tmp68 + tmp96
        tmp98 = tmp97 - tmp31
        tmp99 = tl.exp(tmp98)
        tmp100 = tmp99 / tmp66
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp100, rmask)
''')
