

# Original file: ./XGLMForCausalLM__85_forward_258.23/XGLMForCausalLM__85_forward_258.23_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/er/cer7r3gxqr223g4xed6i4itmyfr7l2g4k3efwntfxwey74bvgg47.py
# Source Nodes: [dropout, softmax], Original ATen: [aten._softmax, aten.native_dropout]
# dropout => gt, mul_3, mul_4
# softmax => amax, convert_element_type_2, convert_element_type_3, div, exp, sub_1, sum_1
triton_per_fused__softmax_native_dropout_3 = async_compile.triton('triton_per_fused__softmax_native_dropout_3', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i64', 3: '*i1', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_native_dropout_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_native_dropout_3(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = -3.3895313892515355e+38
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, float("-inf"))
    tmp9 = triton_helpers.max2(tmp8, 1)[:, None]
    tmp10 = tmp5 - tmp9
    tmp11 = tl.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.load(in_ptr2 + load_seed_offset)
    tmp17 = r2 + (128*x3)
    tmp18 = tl.rand(tmp16, (tmp17).to(tl.uint32))
    tmp19 = tmp18.to(tl.float32)
    tmp20 = 0.1
    tmp21 = tmp19 > tmp20
    tmp22 = tmp11 / tmp15
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21.to(tl.float32)
    tmp25 = tmp24 * tmp23
    tmp26 = 1.1111111111111112
    tmp27 = tmp25 * tmp26
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp21, rmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp23, rmask)
    tl.store(out_ptr5 + (r2 + (128*x3)), tmp27, rmask)
''')
