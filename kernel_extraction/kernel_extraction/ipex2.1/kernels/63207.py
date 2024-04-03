

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/2z/c2za5uxnqx3puu3q63ssj67m3m7b2jox45nvduijk4hwmmxnexph.py
# Source Nodes: [add_56, add_59, l__self___decoder_dropout, mul_80, mul_83, mul_85], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_dropout, aten.sum]
# add_56 => add_66
# add_59 => add_70
# l__self___decoder_dropout => mul_148, mul_149
# mul_80 => mul_150
# mul_83 => mul_157
# mul_85 => mul_163
triton_red_fused__to_copy_add_mul_native_dropout_sum_39 = async_compile.triton('triton_red_fused__to_copy_add_mul_native_dropout_sum_39', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i1', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp16', 8: '*fp32', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_native_dropout_sum_39', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_native_dropout_sum_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp43 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first')
        tmp7 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr5 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr6 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr7 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr8 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr9 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr10 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr11 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp39 = tl.load(in_ptr12 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = 1.1111111111111112
        tmp10 = tmp8 * tmp9
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp13 + tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp4 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp13 * tmp25
        tmp27 = tmp24 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask, tmp30, _tmp29)
        tmp32 = tmp31.to(tl.float32)
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 + tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp35 + tmp37
        tmp40 = tmp10 * tmp39
        tmp41 = tmp38 * tmp40
        tmp42 = tl.broadcast_to(tmp41, [XBLOCK, RBLOCK])
        tmp44 = _tmp43 + tmp42
        _tmp43 = tl.where(rmask, tmp44, _tmp43)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, None)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp29, None)
    tmp43 = tl.sum(_tmp43, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp43, None)
''')
