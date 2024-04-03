

# Original file: ./DebertaForMaskedLM__0_backward_135.1/DebertaForMaskedLM__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/m4/cm4himvr67nddbvny2rln4mzzh5sxkf5pm3d3pwni74xlznqb4wj.py
# Source Nodes: [iadd, sub, trampoline_autograd_apply, truediv], Original ATen: [aten._to_copy, aten.add, aten.div, aten.masked_fill, aten.mul, aten.sub, aten.sum]
# iadd => add
# sub => sub
# trampoline_autograd_apply => full_default_1
# truediv => div
triton_red_fused__to_copy_add_div_masked_fill_mul_sub_sum_30 = async_compile.triton('triton_red_fused__to_copy_add_div_masked_fill_mul_sub_sum_30', '''
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_masked_fill_mul_sub_sum_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_div_masked_fill_mul_sub_sum_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first')
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr4 + (x0 + (768*((r2 + (128*x1)) % 512))), rmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr5 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr6 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = 0.0
        tmp6 = tl.where(tmp0, tmp5, tmp4)
        tmp7 = 1.1111111111111112
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 / tmp17
        tmp19 = tmp8 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp21, None)
''')