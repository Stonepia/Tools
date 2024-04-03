

# Original file: ./T5ForConditionalGeneration__0_backward_171.1/T5ForConditionalGeneration__0_backward_171.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/rt/crtsdhqeai4lel4eaqhrrnzvybgomuyfr6zjnui2zddeyt6ysbde.py
# Source Nodes: [add_63, add_65, add_67, mul_65, mul_67, mul_69], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_dropout_backward, aten.sum]
# add_63 => add_81
# add_65 => add_84
# add_67 => add_86
# mul_65 => mul_183
# mul_67 => mul_189
# mul_69 => mul_195
triton_red_fused__to_copy_add_mul_native_dropout_backward_sum_24 = async_compile.triton('triton_red_fused__to_copy_add_mul_native_dropout_backward_sum_24', '''
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
    meta={'signature': {0: '*bf16', 1: '*i1', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*bf16', 8: '*fp32', 9: '*bf16', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_native_dropout_backward_sum_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_native_dropout_backward_sum_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp39 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first')
        tmp9 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr4 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr5 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr6 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr7 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr8 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr9 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr10 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = 0.04419417382415922
        tmp3 = tmp1 * tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = 1.1111111111111112
        tmp7 = tmp5 * tmp6
        tmp8 = tmp3 * tmp7
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp9 + tmp11
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp12 + tmp14
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 + tmp17
        tmp20 = tmp18 * tmp19
        tmp21 = tmp8 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask, tmp24, _tmp23)
        tmp26 = tmp25.to(tl.float32)
        tmp28 = tmp15 * tmp27
        tmp29 = tmp26 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask, tmp32, _tmp31)
        tmp34 = tmp33.to(tl.float32)
        tmp36 = tmp12 * tmp35
        tmp37 = tmp34 * tmp36
        tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
        tmp40 = _tmp39 + tmp38
        _tmp39 = tl.where(rmask, tmp40, _tmp39)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, None)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp31, None)
    tmp39 = tl.sum(_tmp39, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp39, None)
''')
