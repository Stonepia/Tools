

# Original file: ./OPTForCausalLM__43_backward_180.20/OPTForCausalLM__43_backward_180.20.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/67/c67wxyhtumkzi33vknct2454av2pe64yvy3ndzlvctpgd36eyw2l.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.div, aten.masked_fill, aten.threshold_backward, aten.where]

triton_red_fused__softmax_backward_data_div_masked_fill_threshold_backward_where_9 = async_compile.triton('triton_red_fused__softmax_backward_data_div_masked_fill_threshold_backward_where_9', '''
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
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*i1', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_div_masked_fill_threshold_backward_where_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_backward_data_div_masked_fill_threshold_backward_where_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first')
        tmp7 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first')
        tmp8 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tmp9 * tmp4
        tmp12 = tmp10 - tmp11
        tmp13 = 2.0
        tmp14 = tmp12 / tmp13
        tmp15 = tl.where(tmp7, tmp14, tmp12)
        tmp16 = 0.0
        tmp17 = tl.where(tmp6, tmp16, tmp15)
        tl.store(out_ptr1 + (r1 + (2048*x0)), tmp17, rmask)
''')
