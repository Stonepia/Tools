

# Original file: ./MobileBertForMaskedLM__0_forward_280.0/MobileBertForMaskedLM__0_forward_280.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/tf/ctfomnqzzmpnriastl5fvwyrgikfwrbymjkd4rd5c4o7onlxnh7t.py
# Source Nodes: [cross_entropy, iadd], Original ATen: [aten._log_softmax, aten._to_copy, aten.view]
# cross_entropy => amax_24, convert_element_type_994, exp_24, log, sub_26, sub_27, sum_25
# iadd => view_967
triton_red_fused__log_softmax__to_copy_view_32 = async_compile.triton('triton_red_fused__log_softmax__to_copy_view_32', '''
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
    size_hints=[16384, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_view_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_view_32(in_out_ptr0, in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 30522
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r1 + (30522*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr0 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tl.store(in_out_ptr0 + (r1 + (30522*x0)), tmp4, rmask)
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_out_ptr0 + (r1 + (30522*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 - tmp7
        tmp12 = tl.exp(tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_out_ptr0 + (r1 + (30522*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 - tmp7
        tmp19 = tl.log(tmp14)
        tmp20 = tmp18 - tmp19
        tl.store(out_ptr2 + (r1 + (30522*x0)), tmp20, rmask)
''')
