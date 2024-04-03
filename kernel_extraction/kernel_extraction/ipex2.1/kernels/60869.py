

# Original file: ./OPTForCausalLM__46_backward_179.19/OPTForCausalLM__46_backward_179.19_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/2n/c2nskkwmhuobwfjwjv7czt4xjdbumbgniiricri4emrr423blcqx.py
# Source Nodes: [add, tensor], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.eq, aten.lift_fresh, aten.lt, aten.masked_fill, aten.threshold_backward, aten.where]
# add => add_2
# tensor => full_default
triton_red_fused__softmax_backward_data_add_div_eq_lift_fresh_lt_masked_fill_threshold_backward_where_10 = async_compile.triton('triton_red_fused__softmax_backward_data_add_div_eq_lift_fresh_lt_masked_fill_threshold_backward_where_10', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_add_div_eq_lift_fresh_lt_masked_fill_threshold_backward_where_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_backward_data_add_div_eq_lift_fresh_lt_masked_fill_threshold_backward_where_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    x2 = xindex % 2048
    x4 = (xindex // 24576)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r1 + (2048*x2) + (4194304*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = -3.3895313892515355e+38
        tmp12 = tmp10 < tmp11
        tmp13 = tmp10 == tmp11
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 * tmp17
        tmp19 = tmp17 * tmp6
        tmp20 = tmp18 - tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 2.0
        tmp23 = tmp21 / tmp22
        tmp24 = tl.where(tmp13, tmp23, tmp21)
        tmp25 = 0.0
        tmp26 = tl.where(tmp12, tmp25, tmp24)
        tl.store(out_ptr1 + (r1 + (2048*x0)), tmp26, rmask)
''')
