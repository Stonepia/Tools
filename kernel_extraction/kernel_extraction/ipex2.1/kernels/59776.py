

# Original file: ./T5ForConditionalGeneration__0_backward_171.1/T5ForConditionalGeneration__0_backward_171.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/aq/caqln7l3muqtwjuuezb5grfqjhjzeielxbty56s56ssl3qt2zh72.py
# Source Nodes: [add_32, add_35, l__self___decoder_dropout, mul_32, mul_35, mul_37], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_dropout, aten.sum]
# add_32 => add_40
# add_35 => add_44
# l__self___decoder_dropout => mul_84, mul_85
# mul_32 => mul_86
# mul_35 => mul_93
# mul_37 => mul_99
triton_red_fused__to_copy_add_mul_native_dropout_sum_34 = async_compile.triton('triton_red_fused__to_copy_add_mul_native_dropout_sum_34', '''
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
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp16', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_native_dropout_sum_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_native_dropout_sum_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp40 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first')
        tmp4 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr5 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr6 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr8 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr9 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr10 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr11 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = 1.1111111111111112
        tmp7 = tmp5 * tmp6
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 + tmp9
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp15 = tmp13 * tmp14
        tmp16 = tmp1 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp10 * tmp22
        tmp24 = tmp21 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
        tmp29 = tmp28.to(tl.float32)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 + tmp31
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 + tmp34
        tmp37 = tmp7 * tmp36
        tmp38 = tmp35 * tmp37
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
        tmp41 = _tmp40 + tmp39
        _tmp40 = tl.where(rmask, tmp41, _tmp40)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, None)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp26, None)
    tmp40 = tl.sum(_tmp40, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp40, None)
''')
