

# Original file: ./OPTForCausalLM__49_forward_152.11/OPTForCausalLM__49_forward_152.11.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/6a/c6as4cu42b5osfdqcka5v4gbkzqr6velyk6cd6gcloy53ezhqs4j.py
# Source Nodes: [add, dropout, softmax, tensor], Original ATen: [aten._softmax, aten.add, aten.clone, aten.eq, aten.lift_fresh, aten.lt]
# add => add_2
# dropout => clone_3
# softmax => amax, div, exp, sub_1, sum_1
# tensor => full_default
triton_red_fused__softmax_add_clone_eq_lift_fresh_lt_3 = async_compile.triton('triton_red_fused__softmax_add_clone_eq_lift_fresh_lt_3', '''
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
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*i1', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_clone_eq_lift_fresh_lt_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_add_clone_eq_lift_fresh_lt_3(in_ptr0, in_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp6 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (2048*x0) + (4194304*(x1 // 12))), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = -3.4028234663852886e+38
        tmp4 = triton_helpers.maximum(tmp2, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = triton_helpers.maximum(_tmp6, tmp5)
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = triton_helpers.max2(_tmp6, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr1 + (r2 + (2048*x0) + (4194304*(x1 // 12))), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 + tmp9
        tmp11 = -3.4028234663852886e+38
        tmp12 = triton_helpers.maximum(tmp10, tmp11)
        tmp13 = tmp12 - tmp6
        tmp14 = tl.exp(tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    x5 = (xindex // 24576)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp18 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr1 + (r2 + (2048*x0) + (4194304*(x1 // 12))), rmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr1 + (r2 + (2048*x0) + (4194304*x5)), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp18 + tmp19
        tmp21 = -3.4028234663852886e+38
        tmp22 = triton_helpers.maximum(tmp20, tmp21)
        tmp23 = tmp22 - tmp6
        tmp24 = tl.exp(tmp23)
        tmp25 = tmp24 / tmp16
        tmp27 = tmp18 + tmp26
        tmp28 = tmp27 == tmp21
        tmp29 = tmp27 < tmp21
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp25, rmask)
        tl.store(out_ptr3 + (r2 + (2048*x3)), tmp25, rmask)
        tl.store(out_ptr4 + (r2 + (2048*x3)), tmp28, rmask)
        tl.store(out_ptr5 + (r2 + (2048*x3)), tmp29, rmask)
''')
