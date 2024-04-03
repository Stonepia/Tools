

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/qs/cqszwzb5gmtw763bh6gbbksrdtd3g545unqwnndf5fispipuwoyz.py
# Source Nodes: [add_103, add_97, add_99, mul_138, mul_140, mul_147], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
# add_103 => add_124
# add_97 => add_117
# add_99 => add_120
# mul_138 => mul_272
# mul_140 => mul_278
# mul_147 => mul_289
triton_red_fused__to_copy_add_mul_sum_34 = async_compile.triton('triton_red_fused__to_copy_add_mul_sum_34', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*fp32', 8: '*bf16', 9: '*bf16', 10: '*fp32', 11: '*bf16', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sum_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp33 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
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
        tmp26 = tl.load(in_ptr9 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr10 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr11 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr12 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 + tmp27
        tmp30 = tmp14 * tmp29
        tmp31 = tmp28 * tmp30
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(rmask, tmp34, _tmp33)
        tmp36 = tmp35.to(tl.float32)
        tmp38 = tmp11 * tmp37
        tmp39 = tmp36 * tmp38
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask, tmp42, _tmp41)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, None)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp33, None)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp41, None)
''')
