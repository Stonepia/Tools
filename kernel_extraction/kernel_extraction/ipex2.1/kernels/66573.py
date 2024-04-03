

# Original file: ./speech_transformer__22_inference_62.2/speech_transformer__22_inference_62.2_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/tp/ctp4ishkepxyyase67f7qfnzmlztyolmwn7j45ilrzmtvzs77sfi.py
# Source Nodes: [l__self___layer_stack_0_slf_attn_attention_softmax, masked_fill, repeat, truediv], Original ATen: [aten._softmax, aten.div, aten.masked_fill, aten.repeat]
# l__self___layer_stack_0_slf_attn_attention_softmax => amax, div_1, exp, sub_1, sum_1
# masked_fill => full_default, where
# repeat => repeat
# truediv => div
triton_red_fused__softmax_div_masked_fill_repeat_2 = async_compile.triton('triton_red_fused__softmax_div_masked_fill_repeat_2', '''
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
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp32', 2: 'fp64', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_div_masked_fill_repeat_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_div_masked_fill_repeat_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16320
    rnumel = 204
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 204)
    x3 = xindex
    tmp2 = in_ptr2
    _tmp8 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (204*(x1 % 10))), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r2 + (204*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 / tmp3
        tmp5 = float("-inf")
        tmp6 = tl.where(tmp0, tmp5, tmp4)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp10 = tl.load(in_ptr0 + (r2 + (204*(x1 % 10))), rmask & xmask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr1 + (r2 + (204*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp2.to(tl.float32)
        tmp13 = tmp11 / tmp12
        tmp14 = float("-inf")
        tmp15 = tl.where(tmp10, tmp14, tmp13)
        tmp16 = tmp15 - tmp8
        tmp17 = tl.exp(tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp21 = tl.load(in_ptr0 + (r2 + (204*(x1 % 10))), rmask & xmask, eviction_policy='evict_last')
        tmp22 = tl.load(in_ptr1 + (r2 + (204*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tmp2.to(tl.float32)
        tmp24 = tmp22 / tmp23
        tmp25 = float("-inf")
        tmp26 = tl.where(tmp21, tmp25, tmp24)
        tmp27 = tmp26 - tmp8
        tmp28 = tl.exp(tmp27)
        tmp29 = tmp28 / tmp19
        tl.store(out_ptr2 + (r2 + (204*x3)), tmp29, rmask & xmask)
''')
