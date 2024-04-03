

# Original file: ./YituTechConvBert__0_backward_207.1/YituTechConvBert__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/la/clazncus4znzzd77yurf7utdr3ahrnldugb6mbo6daelk6rx56ia.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten._to_copy, aten.div, aten.native_dropout_backward]

triton_red_fused__softmax_backward_data__to_copy_div_native_dropout_backward_21 = async_compile.triton('triton_red_fused__softmax_backward_data__to_copy_div_native_dropout_backward_21', '''
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
    meta={'signature': {0: '*bf16', 1: '*i1', 2: '*fp32', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data__to_copy_div_native_dropout_backward_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_backward_data__to_copy_div_native_dropout_backward_21(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (512*x1) + (262144*x0) + (1572864*x2)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (6*r3) + (3072*x4)), rmask, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr2 + (x0 + (6*r3) + (3072*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 1.1111111111111112
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp12 = tl.load(in_ptr0 + (r3 + (512*x1) + (262144*x0) + (1572864*x2)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (x0 + (6*r3) + (3072*x4)), rmask, eviction_policy='evict_first')
        tmp19 = tl.load(in_ptr2 + (x0 + (6*r3) + (3072*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = 1.1111111111111112
        tmp17 = tmp15 * tmp16
        tmp18 = tmp13 * tmp17
        tmp20 = tmp18 * tmp19
        tmp21 = tmp19 * tmp10
        tmp22 = tmp20 - tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = 8.0
        tmp25 = tmp23 / tmp24
        tl.store(out_ptr1 + (r3 + (512*x1) + (262144*x0) + (1572864*x2)), tmp25, rmask)
''')
