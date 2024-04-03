

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/gy/cgycr6jd5jfin4hncrmu4j5tkzjreltbkufjvlj4wvkp5nfdi5ya.py
# Source Nodes: [add_4, add_8, l__self___encoder_dropout, mul_1, mul_12, mul_5], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_dropout, aten.sum]
# add_4 => add_6
# add_8 => add_10
# l__self___encoder_dropout => mul_1, mul_2
# mul_1 => mul_3
# mul_12 => mul_22
# mul_5 => mul_11
triton_red_fused__to_copy_add_mul_native_dropout_sum_49 = async_compile.triton('triton_red_fused__to_copy_add_mul_native_dropout_sum_49', '''
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*i1', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: '*fp32', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: 'i32', 19: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_native_dropout_sum_49', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_native_dropout_sum_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp35 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp49 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first')
        tmp10 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr6 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr8 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr9 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr10 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr11 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp39 = tl.load(in_ptr12 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp42 = tl.load(in_ptr13 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp45 = tl.load(in_ptr14 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 + tmp6
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = 1.1111111111111112
        tmp13 = tmp11 * tmp12
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp13 + tmp15
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 + tmp18
        tmp21 = tmp19 * tmp20
        tmp22 = tmp7 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
        tmp27 = tmp26.to(tl.float32)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp27 + tmp29
        tmp32 = tmp16 * tmp31
        tmp33 = tmp30 * tmp32
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(rmask, tmp36, _tmp35)
        tmp38 = tmp37.to(tl.float32)
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp38 + tmp40
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp41 + tmp43
        tmp46 = tmp13 * tmp45
        tmp47 = tmp44 * tmp46
        tmp48 = tl.broadcast_to(tmp47, [XBLOCK, RBLOCK])
        tmp50 = _tmp49 + tmp48
        _tmp49 = tl.where(rmask, tmp50, _tmp49)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp24, None)
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp35, None)
    tmp49 = tl.sum(_tmp49, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp49, None)
''')
