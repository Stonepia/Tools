

# Original file: ./YituTechConvBert__0_backward_171.1/YituTechConvBert__0_backward_171.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/yo/cyo2jekprdek35wvrca4qlf5wucqsjg2ma4ymxrssgj5xyl7qsmq.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.native_dropout_backward]

triton_red_fused__softmax_backward_data_div_native_dropout_backward_16 = async_compile.triton('triton_red_fused__softmax_backward_data_div_native_dropout_backward_16', '''
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
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_div_native_dropout_backward_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_backward_data_div_native_dropout_backward_16(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 6
    x1 = (xindex // 6) % 512
    x2 = (xindex // 3072)
    x4 = (xindex // 6)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (512*x1) + (262144*x0) + (1572864*x2)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (6*r3) + (3072*x4)), rmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr2 + (x0 + (6*r3) + (3072*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = 1.1111111111111112
        tmp4 = tmp2 * tmp3
        tmp5 = tmp0 * tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp13 = tl.load(in_ptr0 + (r3 + (512*x1) + (262144*x0) + (1572864*x2)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (x0 + (6*r3) + (3072*x4)), rmask, eviction_policy='evict_first')
        tmp20 = tl.load(in_ptr2 + (x0 + (6*r3) + (3072*x4)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = 1.1111111111111112
        tmp17 = tmp15 * tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp19 * tmp21
        tmp23 = tmp21 * tmp11
        tmp24 = tmp22 - tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = 8.0
        tmp27 = tmp25 / tmp26
        tl.store(out_ptr1 + (r3 + (512*x1) + (262144*x0) + (1572864*x2)), tmp27, rmask)
''')
