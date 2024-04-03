

# Original file: ./hf_T5___60.0/hf_T5___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/aq/caqjymgws2n4tjopr2r7zb5fdy4gwuli5cxb4k3b5rq6sz4vica2.py
# Source Nodes: [float_9, softmax_6, type_as_6], Original ATen: [aten._softmax, aten._to_copy]
# float_9 => convert_element_type_49
# softmax_6 => amax_6, div_10, exp_6, sub_11, sum_7
# type_as_6 => convert_element_type_50
triton_red_fused__softmax__to_copy_2 = async_compile.triton('triton_red_fused__softmax__to_copy_2', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_2(in_ptr0, in_ptr1, out_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp33 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp19 = tl.load(in_ptr1 + (x1 + (8*tmp18)), None, eviction_policy='evict_last').to(tl.float32)
        tmp20 = r2
        tmp21 = x0
        tmp22 = tmp20 <= tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = 1.0
        tmp26 = tmp25 - tmp24
        tmp27 = -3.3895313892515355e+38
        tmp28 = tmp26 * tmp27
        tmp29 = tmp19 + tmp28
        tmp30 = tmp0 + tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = triton_helpers.maximum(_tmp33, tmp32)
        _tmp33 = tl.where(rmask, tmp34, _tmp33)
    tmp33 = triton_helpers.max2(_tmp33, 1)[:, None]
    _tmp70 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp35 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = (-1)*(tl.minimum(0, r2 + ((-1)*x0), tl.PropagateNan.NONE))
        tmp37 = tl.full([1, 1], 16, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp36.to(tl.float32)
        tmp40 = 16.0
        tmp41 = tmp39 / tmp40
        tmp42 = tl.log(tmp41)
        tmp43 = 2.0794415416798357
        tmp44 = tmp42 / tmp43
        tmp45 = tmp44 * tmp40
        tmp46 = tmp45.to(tl.int64)
        tmp47 = tmp46 + tmp37
        tmp48 = tl.full([1, 1], 31, tl.int64)
        tmp49 = triton_helpers.minimum(tmp47, tmp48)
        tmp50 = tl.where(tmp38, tmp36, tmp49)
        tmp51 = tl.full([1, 1], 0, tl.int64)
        tmp52 = tmp50 + tmp51
        tmp53 = tl.where(tmp52 < 0, tmp52 + 32, tmp52)
        # tl.device_assert((0 <= tmp53) & (tmp53 < 32), "index out of bounds: 0 <= tmp53 < 32")
        tmp54 = tl.load(in_ptr1 + (x1 + (8*tmp53)), None, eviction_policy='evict_first').to(tl.float32)
        tmp55 = r2
        tmp56 = x0
        tmp57 = tmp55 <= tmp56
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tmp58.to(tl.float32)
        tmp60 = 1.0
        tmp61 = tmp60 - tmp59
        tmp62 = -3.3895313892515355e+38
        tmp63 = tmp61 * tmp62
        tmp64 = tmp54 + tmp63
        tmp65 = tmp35 + tmp64
        tmp66 = tmp65.to(tl.float32)
        tmp67 = tmp66 - tmp33
        tmp68 = tl.exp(tmp67)
        tmp69 = tl.broadcast_to(tmp68, [XBLOCK, RBLOCK])
        tmp71 = _tmp70 + tmp69
        _tmp70 = tl.where(rmask, tmp71, _tmp70)
        tl.store(out_ptr1 + (r2 + (2048*x3)), tmp67, rmask)
    tmp70 = tl.sum(_tmp70, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp72 = tl.load(out_ptr1 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp73 = tl.exp(tmp72)
        tmp74 = tmp73 / tmp70
        tmp75 = tmp74.to(tl.float32)
        tl.store(out_ptr3 + (r2 + (2048*x3)), tmp75, rmask)
''')
