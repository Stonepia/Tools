

# Original file: ./eca_halonext26ts___60.0/eca_halonext26ts___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/5b/c5bwrycfv2du7bifbfhgbmwvzsrmzsxmwywcoo2jm5ymscryjqn3.py
# Source Nodes: [add_9, mul_6, softmax_1], Original ATen: [aten._softmax, aten.add, aten.mul]
# add_9 => add_59
# mul_6 => mul_103
# softmax_1 => amax_1, convert_element_type_121, convert_element_type_122, div_1, exp_1, sub_26, sum_2
triton_red_fused__softmax_add_mul_36 = async_compile.triton('triton_red_fused__softmax_add_mul_36', '''
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
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_36', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_add_mul_36(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp25 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = 0.25
        tmp2 = tmp0 * tmp1
        tmp3 = 11 + (23*(x0 // 4)) + (r2 // 12)
        tmp4 = tl.full([1, 1], 96, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (11 + (23*(x0 // 4)) + (r2 // 12)) % 24
        tmp7 = tl.full([1, 1], 23, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 4)) + (r2 // 12)) // 24)) + (92*(x0 % 4)) + (368*((((4*x1) + (x0 % 4)) // 4) % 4096)) + ((11 + (23*(x0 // 4)) + (r2 // 12)) % 24)), rmask & tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tl.where(tmp9, tmp10, 0.0)
        tmp12 = tl.where(tmp5, tmp11, 0.0)
        tmp13 = 11 + (23*(x0 % 4)) + (r2 % 12)
        tmp14 = tmp13 < tmp4
        tmp15 = (11 + (23*(x0 % 4)) + (r2 % 12)) % 24
        tmp16 = tmp15 < tmp7
        tmp17 = tmp16 & tmp14
        tmp18 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 4)) + (r2 % 12)) // 24) % 4)) + (92*(x0 // 4)) + (368*x1) + ((11 + (23*(x0 % 4)) + (r2 % 12)) % 24)), rmask & tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tl.where(tmp17, tmp18, 0.0)
        tmp20 = tl.where(tmp14, tmp19, 0.0)
        tmp21 = tmp12 + tmp20
        tmp22 = tmp2 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = triton_helpers.maximum(_tmp25, tmp24)
        _tmp25 = tl.where(rmask, tmp26, _tmp25)
    tmp25 = triton_helpers.max2(_tmp25, 1)[:, None]
    _tmp54 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp27 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = 0.25
        tmp29 = tmp27 * tmp28
        tmp30 = 11 + (23*(x0 // 4)) + (r2 // 12)
        tmp31 = tl.full([1, 1], 96, tl.int64)
        tmp32 = tmp30 < tmp31
        tmp33 = (11 + (23*(x0 // 4)) + (r2 // 12)) % 24
        tmp34 = tl.full([1, 1], 23, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = tmp35 & tmp32
        tmp37 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 4)) + (r2 // 12)) // 24)) + (92*(x0 % 4)) + (368*((((4*x1) + (x0 % 4)) // 4) % 4096)) + ((11 + (23*(x0 // 4)) + (r2 // 12)) % 24)), rmask & tmp36, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp38 = tl.where(tmp36, tmp37, 0.0)
        tmp39 = tl.where(tmp32, tmp38, 0.0)
        tmp40 = 11 + (23*(x0 % 4)) + (r2 % 12)
        tmp41 = tmp40 < tmp31
        tmp42 = (11 + (23*(x0 % 4)) + (r2 % 12)) % 24
        tmp43 = tmp42 < tmp34
        tmp44 = tmp43 & tmp41
        tmp45 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 4)) + (r2 % 12)) // 24) % 4)) + (92*(x0 // 4)) + (368*x1) + ((11 + (23*(x0 % 4)) + (r2 % 12)) % 24)), rmask & tmp44, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp46 = tl.where(tmp44, tmp45, 0.0)
        tmp47 = tl.where(tmp41, tmp46, 0.0)
        tmp48 = tmp39 + tmp47
        tmp49 = tmp29 + tmp48
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tmp50 - tmp25
        tmp52 = tl.exp(tmp51)
        tmp53 = tl.broadcast_to(tmp52, [XBLOCK, RBLOCK])
        tmp55 = _tmp54 + tmp53
        _tmp54 = tl.where(rmask, tmp55, _tmp54)
        tl.store(out_ptr1 + (r2 + (144*x3)), tmp52, rmask)
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp56 = tl.load(out_ptr1 + (r2 + (144*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp57 = tmp56 / tmp54
        tmp58 = tmp57.to(tl.float32)
        tl.store(out_ptr3 + (r2 + (144*x3)), tmp58, rmask)
''')