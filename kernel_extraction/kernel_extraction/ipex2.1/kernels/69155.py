

# Original file: ./XGLMForCausalLM__85_backward_284.30/XGLMForCausalLM__85_backward_284.30_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/22/c22ojryvvxizrwz4fsivkdcxj5dvzuuwkj4acd355dr4ui5ytds3.py
# Source Nodes: [add, tensor], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.eq, aten.lift_fresh, aten.lt, aten.masked_fill, aten.native_dropout_backward, aten.where]
# add => add_2
# tensor => full_default
triton_per_fused__softmax_backward_data_add_div_eq_lift_fresh_lt_masked_fill_native_dropout_backward_where_9 = async_compile.triton('triton_per_fused__softmax_backward_data_add_div_eq_lift_fresh_lt_masked_fill_native_dropout_backward_where_9', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i1', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_add_div_eq_lift_fresh_lt_masked_fill_native_dropout_backward_where_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_backward_data_add_div_eq_lift_fresh_lt_masked_fill_native_dropout_backward_where_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r1 + (128*x2)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp16 = tmp14 + tmp15
    tmp17 = -3.3895313892515355e+38
    tmp18 = tmp16 == tmp17
    tmp19 = tmp8 * tmp13
    tmp20 = tmp9 - tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 2.0
    tmp23 = tmp21 / tmp22
    tmp24 = tl.where(tmp18, tmp23, tmp21)
    tmp25 = tmp16 < tmp17
    tmp26 = 0.0
    tmp27 = tl.where(tmp25, tmp26, tmp24)
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp27, rmask)
''')
