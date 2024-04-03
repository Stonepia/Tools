

# Original file: ./XGLMForCausalLM__88_backward_283.29/XGLMForCausalLM__88_backward_283.29_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/2o/c2oy3l42jlif2q2myuykxaioutm66kmp5tsjkorrmvy72rk4qv4u.py
# Source Nodes: [add, tensor], Original ATen: [aten._softmax_backward_data, aten._to_copy, aten.add, aten.div, aten.eq, aten.lift_fresh, aten.lt, aten.masked_fill, aten.native_dropout_backward, aten.where]
# add => add_2
# tensor => full_default
triton_per_fused__softmax_backward_data__to_copy_add_div_eq_lift_fresh_lt_masked_fill_native_dropout_backward_where_9 = async_compile.triton('triton_per_fused__softmax_backward_data__to_copy_add_div_eq_lift_fresh_lt_masked_fill_native_dropout_backward_where_9', '''
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data__to_copy_add_div_eq_lift_fresh_lt_masked_fill_native_dropout_backward_where_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_backward_data__to_copy_add_div_eq_lift_fresh_lt_masked_fill_native_dropout_backward_where_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r1 + (128*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp14 + tmp15
    tmp17 = -3.4028234663852886e+38
    tmp18 = tmp16 < tmp17
    tmp19 = tmp16 == tmp17
    tmp20 = tmp7 * tmp12
    tmp21 = tmp8 - tmp20
    tmp22 = 2.0
    tmp23 = tmp21 / tmp22
    tmp24 = tl.where(tmp19, tmp23, tmp21)
    tmp25 = 0.0
    tmp26 = tl.where(tmp18, tmp25, tmp24)
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask)
''')
