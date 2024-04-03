

# Original file: ./XLNetLMHeadModel__0_forward_565.0/XLNetLMHeadModel__0_forward_565.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/xm/cxms6l7rtvc26ycqit4ms27cfb2do335jj5kifuomvjulsvclez6.py
# Source Nodes: [add_2, add_3, index_select, l__mod___transformer_layer_0_rel_attn_dropout, mul, softmax], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
# add_2 => add_2
# add_3 => add_3
# index_select => index
# l__mod___transformer_layer_0_rel_attn_dropout => gt_2, mul_7, mul_8
# mul => mul_6
# softmax => amax, div_1, exp, sub, sum_1
triton_red_fused__softmax_add_index_select_mul_native_dropout_6 = async_compile.triton('triton_red_fused__softmax_add_index_select_mul_native_dropout_6', '''
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
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_index_select_mul_native_dropout_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_add_index_select_mul_native_dropout_6(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512) % 16
    x2 = (xindex // 8192)
    _tmp8 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = 0.125
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp10 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = 0.0
        tmp14 = tmp12 + tmp13
        tmp15 = 0.125
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = tl.exp(tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask, tmp21, _tmp20)
        tmp22 = tl.load(in_ptr2 + load_seed_offset)
        tmp23 = r3 + (512*x4)
        tmp24 = tl.rand(tmp22, (tmp23).to(tl.uint32))
        tl.store(out_ptr2 + (r3 + (512*x4)), tmp24, rmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp25 = tl.load(out_ptr2 + (r3 + (512*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_first', other=0.0)
        tmp26 = 0.1
        tmp27 = tmp25 > tmp26
        tmp30 = tmp28 + tmp29
        tmp31 = 0.0
        tmp32 = tmp30 + tmp31
        tmp33 = 0.125
        tmp34 = tmp32 * tmp33
        tmp35 = tmp34 - tmp8
        tmp36 = tl.exp(tmp35)
        tmp37 = tmp36 / tmp20
        tmp38 = tmp27.to(tl.float32)
        tmp39 = tmp38 * tmp37
        tmp40 = 1.1111111111111112
        tmp41 = tmp39 * tmp40
        tl.store(out_ptr3 + (r3 + (512*x4)), tmp27, rmask)
        tl.store(out_ptr4 + (r3 + (512*x4)), tmp37, rmask)
        tl.store(out_ptr5 + (r3 + (512*x4)), tmp41, rmask)
''')
