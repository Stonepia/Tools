

# Original file: ./hf_T5_generate__53_inference_93.33/hf_T5_generate__53_inference_93.33_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/jo/cjotp56dv77hf6e4o6q5z4dfv2emrxqoao5warh7imfrglasmw7t.py
# Source Nodes: [softmax], Original ATen: [aten._softmax]
# softmax => amax, div_2, exp, sub_8, sum_1
triton_red_fused__softmax_6 = async_compile.triton('triton_red_fused__softmax_6', '''
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
    size_hints=[8, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_6(in_ptr0, in_ptr1, in_ptr2, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp29 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + x0 + (ks0*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = (-1)*(tl.minimum(0, r1 + ((-1)*ks0), tl.PropagateNan.NONE))
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
        tmp19 = tl.load(in_ptr1 + (x0 + (8*tmp18)), xmask, eviction_policy='evict_last')
        tmp21 = 1.0
        tmp22 = tmp20 * tmp21
        tmp23 = tmp21 - tmp22
        tmp24 = -3.4028234663852886e+38
        tmp25 = tmp23 * tmp24
        tmp26 = tmp19 + tmp25
        tmp27 = tmp0 + tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = triton_helpers.maximum(_tmp29, tmp28)
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
    tmp29 = triton_helpers.max2(_tmp29, 1)[:, None]
    _tmp62 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp31 = tl.load(in_ptr0 + (r1 + x0 + (ks0*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = (-1)*(tl.minimum(0, r1 + ((-1)*ks0), tl.PropagateNan.NONE))
        tmp33 = tl.full([1, 1], 16, tl.int64)
        tmp34 = tmp32 < tmp33
        tmp35 = tmp32.to(tl.float32)
        tmp36 = 16.0
        tmp37 = tmp35 / tmp36
        tmp38 = tl.log(tmp37)
        tmp39 = 2.0794415416798357
        tmp40 = tmp38 / tmp39
        tmp41 = tmp40 * tmp36
        tmp42 = tmp41.to(tl.int64)
        tmp43 = tmp42 + tmp33
        tmp44 = tl.full([1, 1], 31, tl.int64)
        tmp45 = triton_helpers.minimum(tmp43, tmp44)
        tmp46 = tl.where(tmp34, tmp32, tmp45)
        tmp47 = tl.full([1, 1], 0, tl.int64)
        tmp48 = tmp46 + tmp47
        tmp49 = tl.where(tmp48 < 0, tmp48 + 32, tmp48)
        # tl.device_assert((0 <= tmp49) & (tmp49 < 32), "index out of bounds: 0 <= tmp49 < 32")
        tmp50 = tl.load(in_ptr1 + (x0 + (8*tmp49)), xmask, eviction_policy='evict_last')
        tmp52 = 1.0
        tmp53 = tmp51 * tmp52
        tmp54 = tmp52 - tmp53
        tmp55 = -3.4028234663852886e+38
        tmp56 = tmp54 * tmp55
        tmp57 = tmp50 + tmp56
        tmp58 = tmp31 + tmp57
        tmp59 = tmp58 - tmp29
        tmp60 = tl.exp(tmp59)
        tmp61 = tl.broadcast_to(tmp60, [XBLOCK, RBLOCK])
        tmp63 = _tmp62 + tmp61
        _tmp62 = tl.where(rmask & xmask, tmp63, _tmp62)
    tmp62 = tl.sum(_tmp62, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp64 = tl.load(in_ptr0 + (r1 + x0 + (ks0*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp84 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp65 = (-1)*(tl.minimum(0, r1 + ((-1)*ks0), tl.PropagateNan.NONE))
        tmp66 = tl.full([1, 1], 16, tl.int64)
        tmp67 = tmp65 < tmp66
        tmp68 = tmp65.to(tl.float32)
        tmp69 = 16.0
        tmp70 = tmp68 / tmp69
        tmp71 = tl.log(tmp70)
        tmp72 = 2.0794415416798357
        tmp73 = tmp71 / tmp72
        tmp74 = tmp73 * tmp69
        tmp75 = tmp74.to(tl.int64)
        tmp76 = tmp75 + tmp66
        tmp77 = tl.full([1, 1], 31, tl.int64)
        tmp78 = triton_helpers.minimum(tmp76, tmp77)
        tmp79 = tl.where(tmp67, tmp65, tmp78)
        tmp80 = tl.full([1, 1], 0, tl.int64)
        tmp81 = tmp79 + tmp80
        tmp82 = tl.where(tmp81 < 0, tmp81 + 32, tmp81)
        # tl.device_assert((0 <= tmp82) & (tmp82 < 32), "index out of bounds: 0 <= tmp82 < 32")
        tmp83 = tl.load(in_ptr1 + (x0 + (8*tmp82)), xmask, eviction_policy='evict_first')
        tmp85 = 1.0
        tmp86 = tmp84 * tmp85
        tmp87 = tmp85 - tmp86
        tmp88 = -3.4028234663852886e+38
        tmp89 = tmp87 * tmp88
        tmp90 = tmp83 + tmp89
        tmp91 = tmp64 + tmp90
        tmp92 = tmp91 - tmp29
        tmp93 = tl.exp(tmp92)
        tmp94 = tmp93 / tmp62
        tl.store(out_ptr2 + (r1 + x0 + (ks0*x0)), tmp94, rmask & xmask)
''')
