

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/zf/czftknku4pou2wt77x6sx2ygs2ocognmt6gtq2sp63xonntet3m4.py
# Source Nodes: [add_18, add_20, add_22, mul_17, mul_19, mul_21, mul_23], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
# add_18 => add_23
# add_20 => add_26
# add_22 => add_28
# mul_17 => mul_47
# mul_19 => mul_53
# mul_21 => mul_59
# mul_23 => mul_65
triton_red_fused__to_copy_add_mul_sum_41 = async_compile.triton('triton_red_fused__to_copy_add_mul_sum_41', '''
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp16', 9: '*fp32', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp32', 14: '*fp16', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: 'i32', 21: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_41', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sum_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp44 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp52 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr5 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr8 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr9 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr10 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp34 = tl.load(in_ptr11 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr12 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp40 = tl.load(in_ptr13 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.load(in_ptr14 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp48 = tl.load(in_ptr15 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 + tmp6
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 + tmp10
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp11 + tmp13
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 + tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = tmp7 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
        tmp25 = tmp24.to(tl.float32)
        tmp27 = tmp14 * tmp26
        tmp28 = tmp25 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
        tmp33 = tmp32.to(tl.float32)
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp33 + tmp35
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp36 + tmp38
        tmp41 = tmp11 * tmp40
        tmp42 = tmp39 * tmp41
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
        tmp45 = _tmp44 + tmp43
        _tmp44 = tl.where(rmask, tmp45, _tmp44)
        tmp47 = tmp46.to(tl.float32)
        tmp49 = tmp8 * tmp48
        tmp50 = tmp47 * tmp49
        tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
        tmp53 = _tmp52 + tmp51
        _tmp52 = tl.where(rmask, tmp53, _tmp52)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, None)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp30, None)
    tmp44 = tl.sum(_tmp44, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp44, None)
    tmp52 = tl.sum(_tmp52, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp52, None)
''')
