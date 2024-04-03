

# Original file: ./XGLMForCausalLM__58_forward_177.14/XGLMForCausalLM__58_forward_177.14_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/w3/cw3mt3szca6gma5gym3jeizgnt4qs445uyliouoffxyd5a5pyhyc.py
# Source Nodes: [bmm_1, dropout, softmax], Original ATen: [aten._softmax, aten._to_copy, aten.native_dropout]
# bmm_1 => convert_element_type_9
# dropout => gt, mul_3, mul_4
# softmax => amax, div, exp, sub_1, sum_1
triton_per_fused__softmax__to_copy_native_dropout_5 = async_compile.triton('triton_per_fused__softmax__to_copy_native_dropout_5', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*i64', 3: '*i1', 4: '*fp32', 5: '*fp16', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_native_dropout_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_native_dropout_5(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (r2 + (128*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = -3.4028234663852886e+38
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
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
    tmp19 = 0.1
    tmp20 = tmp18 > tmp19
    tmp21 = tmp11 / tmp15
    tmp22 = tmp20.to(tl.float32)
    tmp23 = tmp22 * tmp21
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp20, rmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp21, rmask)
    tl.store(out_ptr5 + (r2 + (128*x3)), tmp26, rmask)
''')
