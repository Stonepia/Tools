

# Original file: ./maml__22_forward_66.3/maml__22_forward_66.3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/6g/c6gfbgpnxkfjik2a3h4s2dbmgrtzycjfirumre2ntw6k3u72lhym.py
# Source Nodes: [argmax, eq, softmax, sum_1], Original ATen: [aten._softmax, aten.argmax, aten.eq, aten.sum]
# argmax => argmax
# eq => eq
# softmax => amax, div, exp, sub_22, sum_1
# sum_1 => sum_2
triton_per_fused__softmax_argmax_eq_sum_15 = async_compile.triton('triton_per_fused__softmax_argmax_eq_sum_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[1, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i64', 3: '*i64', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_argmax_eq_sum_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_argmax_eq_sum_15(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 75
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (5*r0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (1 + (5*r0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (2 + (5*r0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr0 + (3 + (5*r0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (4 + (5*r0)), rmask, other=0.0)
    tmp85 = tl.load(in_ptr1 + (r0), rmask, other=0.0)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp0 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tmp1 - tmp8
    tmp12 = tl.exp(tmp11)
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tl.exp(tmp14)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tl.exp(tmp17)
    tmp19 = tmp16 + tmp18
    tmp20 = tmp7 - tmp8
    tmp21 = tl.exp(tmp20)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp10 / tmp22
    tmp24 = tmp12 / tmp22
    tmp25 = tmp23 > tmp24
    tmp26 = tmp23 == tmp24
    tmp27 = tmp23 != tmp23
    tmp28 = tmp24 != tmp24
    tmp29 = tmp27 > tmp28
    tmp30 = tmp25 | tmp29
    tmp31 = tmp27 & tmp28
    tmp32 = tmp26 | tmp31
    tmp33 = tl.full([1, 1], 0, tl.int64)
    tmp34 = tl.full([1, 1], 1, tl.int64)
    tmp35 = tmp33 < tmp34
    tmp36 = tmp32 & tmp35
    tmp37 = tmp30 | tmp36
    tmp38 = tl.where(tmp37, tmp23, tmp24)
    tmp39 = tl.where(tmp37, tmp33, tmp34)
    tmp40 = tmp15 / tmp22
    tmp41 = tmp38 > tmp40
    tmp42 = tmp38 == tmp40
    tmp43 = tmp38 != tmp38
    tmp44 = tmp40 != tmp40
    tmp45 = tmp43 > tmp44
    tmp46 = tmp41 | tmp45
    tmp47 = tmp43 & tmp44
    tmp48 = tmp42 | tmp47
    tmp49 = tl.full([1, 1], 2, tl.int64)
    tmp50 = tmp39 < tmp49
    tmp51 = tmp48 & tmp50
    tmp52 = tmp46 | tmp51
    tmp53 = tl.where(tmp52, tmp38, tmp40)
    tmp54 = tl.where(tmp52, tmp39, tmp49)
    tmp55 = tmp18 / tmp22
    tmp56 = tmp53 > tmp55
    tmp57 = tmp53 == tmp55
    tmp58 = tmp53 != tmp53
    tmp59 = tmp55 != tmp55
    tmp60 = tmp58 > tmp59
    tmp61 = tmp56 | tmp60
    tmp62 = tmp58 & tmp59
    tmp63 = tmp57 | tmp62
    tmp64 = tl.full([1, 1], 3, tl.int64)
    tmp65 = tmp54 < tmp64
    tmp66 = tmp63 & tmp65
    tmp67 = tmp61 | tmp66
    tmp68 = tl.where(tmp67, tmp53, tmp55)
    tmp69 = tl.where(tmp67, tmp54, tmp64)
    tmp70 = tmp21 / tmp22
    tmp71 = tmp68 > tmp70
    tmp72 = tmp68 == tmp70
    tmp73 = tmp68 != tmp68
    tmp74 = tmp70 != tmp70
    tmp75 = tmp73 > tmp74
    tmp76 = tmp71 | tmp75
    tmp77 = tmp73 & tmp74
    tmp78 = tmp72 | tmp77
    tmp79 = tl.full([1, 1], 4, tl.int64)
    tmp80 = tmp69 < tmp79
    tmp81 = tmp78 & tmp80
    tmp82 = tmp76 | tmp81
    tmp83 = tl.where(tmp82, tmp68, tmp70)
    tmp84 = tl.where(tmp82, tmp69, tmp79)
    tmp86 = tmp84 == tmp85
    tmp87 = tmp86.to(tl.int64)
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK, RBLOCK])
    tmp90 = tl.where(rmask, tmp88, 0)
    tmp91 = tl.sum(tmp90, 1)[:, None]
    tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp84, rmask)
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp91, None)
''')
