

# Original file: ./DistillGPT2__0_forward_97.0/DistillGPT2__0_forward_97.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/pw/cpwmxesiut3yp2l3iyaz323r6zbggxpfv6j5rff3swfeozfvyqpp.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax, aten._to_copy]
# cross_entropy => amax_6, convert_element_type_86, exp_6, log, sub_19, sub_20, sum_7
triton_red_fused__log_softmax__to_copy_20 = async_compile.triton('triton_red_fused__log_softmax__to_copy_20', '''
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
    size_hints=[8192, 65536],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_20(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8176
    rnumel = 50257
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50257*(x0 % 511)) + (25731584*(x0 // 511))), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (50257*(x0 % 511)) + (25731584*(x0 // 511))), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = tl.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (50257*(x0 % 511)) + (25731584*(x0 // 511))), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13 - tmp3
        tmp15 = tl.log(tmp10)
        tmp16 = tmp14 - tmp15
        tl.store(out_ptr2 + (r1 + (50257*x0)), tmp16, rmask & xmask)
''')