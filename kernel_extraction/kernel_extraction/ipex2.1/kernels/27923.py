

# Original file: ./maml__23_forward_69.4/maml__23_forward_69.4_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/s3/cs36w7v2kdcdrr5vsofnggizxtprob7pmlr7xy557qdof5ixubjr.py
# Source Nodes: [argmax, eq, softmax, sum_1], Original ATen: [aten._softmax, aten.argmax, aten.eq, aten.sum]
# argmax => argmax
# eq => eq
# softmax => amax, convert_element_type_16, convert_element_type_17, div, exp, sub_4, sum_1
# sum_1 => sum_2
triton_per_fused__softmax_argmax_eq_sum_9 = async_compile.triton('triton_per_fused__softmax_argmax_eq_sum_9', '''
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
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*i64', 3: '*i64', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_argmax_eq_sum_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_argmax_eq_sum_9(in_ptr0, in_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (5*r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (1 + (5*r0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (2 + (5*r0)), rmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (3 + (5*r0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (4 + (5*r0)), rmask, other=0.0).to(tl.float32)
    tmp95 = tl.load(in_ptr1 + (r0), rmask, other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp1 - tmp13
    tmp15 = tl.exp(tmp14)
    tmp16 = tmp3 - tmp13
    tmp17 = tl.exp(tmp16)
    tmp18 = tmp15 + tmp17
    tmp19 = tmp6 - tmp13
    tmp20 = tl.exp(tmp19)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp9 - tmp13
    tmp23 = tl.exp(tmp22)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp12 - tmp13
    tmp26 = tl.exp(tmp25)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp15 / tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp17 / tmp27
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 > tmp31
    tmp33 = tmp29 == tmp31
    tmp34 = tmp29 != tmp29
    tmp35 = tmp31 != tmp31
    tmp36 = tmp34 > tmp35
    tmp37 = tmp32 | tmp36
    tmp38 = tmp34 & tmp35
    tmp39 = tmp33 | tmp38
    tmp40 = tl.full([1, 1], 0, tl.int64)
    tmp41 = tl.full([1, 1], 1, tl.int64)
    tmp42 = tmp40 < tmp41
    tmp43 = tmp39 & tmp42
    tmp44 = tmp37 | tmp43
    tmp45 = tl.where(tmp44, tmp29, tmp31)
    tmp46 = tl.where(tmp44, tmp40, tmp41)
    tmp47 = tmp20 / tmp27
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp45 > tmp48
    tmp50 = tmp45 == tmp48
    tmp51 = tmp45 != tmp45
    tmp52 = tmp48 != tmp48
    tmp53 = tmp51 > tmp52
    tmp54 = tmp49 | tmp53
    tmp55 = tmp51 & tmp52
    tmp56 = tmp50 | tmp55
    tmp57 = tl.full([1, 1], 2, tl.int64)
    tmp58 = tmp46 < tmp57
    tmp59 = tmp56 & tmp58
    tmp60 = tmp54 | tmp59
    tmp61 = tl.where(tmp60, tmp45, tmp48)
    tmp62 = tl.where(tmp60, tmp46, tmp57)
    tmp63 = tmp23 / tmp27
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp61 > tmp64
    tmp66 = tmp61 == tmp64
    tmp67 = tmp61 != tmp61
    tmp68 = tmp64 != tmp64
    tmp69 = tmp67 > tmp68
    tmp70 = tmp65 | tmp69
    tmp71 = tmp67 & tmp68
    tmp72 = tmp66 | tmp71
    tmp73 = tl.full([1, 1], 3, tl.int64)
    tmp74 = tmp62 < tmp73
    tmp75 = tmp72 & tmp74
    tmp76 = tmp70 | tmp75
    tmp77 = tl.where(tmp76, tmp61, tmp64)
    tmp78 = tl.where(tmp76, tmp62, tmp73)
    tmp79 = tmp26 / tmp27
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tmp77 > tmp80
    tmp82 = tmp77 == tmp80
    tmp83 = tmp77 != tmp77
    tmp84 = tmp80 != tmp80
    tmp85 = tmp83 > tmp84
    tmp86 = tmp81 | tmp85
    tmp87 = tmp83 & tmp84
    tmp88 = tmp82 | tmp87
    tmp89 = tl.full([1, 1], 4, tl.int64)
    tmp90 = tmp78 < tmp89
    tmp91 = tmp88 & tmp90
    tmp92 = tmp86 | tmp91
    tmp93 = tl.where(tmp92, tmp77, tmp80)
    tmp94 = tl.where(tmp92, tmp78, tmp89)
    tmp96 = tmp94 == tmp95
    tmp97 = tmp96.to(tl.int64)
    tmp98 = tl.broadcast_to(tmp97, [XBLOCK, RBLOCK])
    tmp100 = tl.where(rmask, tmp98, 0)
    tmp101 = tl.sum(tmp100, 1)[:, None]
    tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp94, rmask)
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp101, None)
''')
