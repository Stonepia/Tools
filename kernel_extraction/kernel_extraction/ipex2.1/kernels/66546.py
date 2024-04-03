

# Original file: ./speech_transformer__22_inference_62.2/speech_transformer__22_inference_62.2_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/js/cjsriyaopfc3bzm6kukthc4b25bozcfds7xt5qab5yymnnamatm3.py
# Source Nodes: [l__self___layer_stack_0_slf_attn_attention_softmax, masked_fill, repeat, truediv], Original ATen: [aten._softmax, aten.div, aten.masked_fill, aten.repeat]
# l__self___layer_stack_0_slf_attn_attention_softmax => amax, convert_element_type_14, convert_element_type_15, div_1, exp, sub_1, sum_1
# masked_fill => full_default, where
# repeat => repeat
# truediv => div
triton_red_fused__softmax_div_masked_fill_repeat_3 = async_compile.triton('triton_red_fused__softmax_div_masked_fill_repeat_3', '''
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
    meta={'signature': {0: '*i1', 1: '*bf16', 2: 'fp64', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_div_masked_fill_repeat_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_div_masked_fill_repeat_3(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16320
    rnumel = 204
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 204)
    x3 = xindex
    tmp2 = in_ptr2
    _tmp9 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (204*(x1 % 10))), rmask & xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r2 + (204*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 / tmp3
        tmp5 = float("-inf")
        tmp6 = tl.where(tmp0, tmp5, tmp4)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = triton_helpers.maximum(_tmp9, tmp8)
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = triton_helpers.max2(_tmp9, 1)[:, None]
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp11 = tl.load(in_ptr0 + (r2 + (204*(x1 % 10))), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr1 + (r2 + (204*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp2.to(tl.float32)
        tmp14 = tmp12 / tmp13
        tmp15 = float("-inf")
        tmp16 = tl.where(tmp11, tmp15, tmp14)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 - tmp9
        tmp19 = tl.exp(tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp23 = tl.load(in_ptr0 + (r2 + (204*(x1 % 10))), rmask & xmask, eviction_policy='evict_last')
        tmp24 = tl.load(in_ptr1 + (r2 + (204*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tmp2.to(tl.float32)
        tmp26 = tmp24 / tmp25
        tmp27 = float("-inf")
        tmp28 = tl.where(tmp23, tmp27, tmp26)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29 - tmp9
        tmp31 = tl.exp(tmp30)
        tmp32 = tmp31 / tmp21
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (204*x3)), tmp33, rmask & xmask)
''')
