

# Original file: ./speech_transformer__28_inference_68.8/speech_transformer__28_inference_68.8_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/yp/cypo4gbkqjci6oj3exouoyuxz5god5kkfcvid7xswjmqae5ughsl.py
# Source Nodes: [l__self___layer_stack_0_slf_attn_attention_softmax, masked_fill, repeat, truediv], Original ATen: [aten._softmax, aten.div, aten.masked_fill, aten.repeat]
# l__self___layer_stack_0_slf_attn_attention_softmax => amax, convert_element_type, convert_element_type_1, div_1, exp, sub, sum_1
# masked_fill => full_default, where
# repeat => repeat
# truediv => div
triton_per_fused__softmax_div_masked_fill_repeat_2 = async_compile.triton('triton_per_fused__softmax_div_masked_fill_repeat_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*bf16', 2: 'fp64', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_div_masked_fill_repeat_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_div_masked_fill_repeat_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1760
    rnumel = 22
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 22
    x1 = (xindex // 22)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (22*x0) + (484*(x1 % 10))), rmask & xmask)
    tmp1 = tl.load(in_ptr1 + (r2 + (22*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp2 = in_ptr2
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 / tmp3
    tmp5 = float("-inf")
    tmp6 = tl.where(tmp0, tmp5, tmp4)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, float("-inf"))
    tmp11 = triton_helpers.max2(tmp10, 1)[:, None]
    tmp12 = tmp7 - tmp11
    tmp13 = tl.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tmp13 / tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (22*x3)), tmp19, rmask & xmask)
''')
