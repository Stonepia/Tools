

# Original file: ./XGLMForCausalLM__37_forward_114.7/XGLMForCausalLM__37_forward_114.7_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/eg/cegr72axjqnl56nxu4zqiiylxurpj7gem2ohtuulccpfnztq2afq.py
# Source Nodes: [add, dropout, softmax, tensor], Original ATen: [aten._softmax, aten.add, aten.eq, aten.lift_fresh, aten.lt, aten.native_dropout]
# add => add_2
# dropout => gt, mul_3, mul_4
# softmax => amax, div, exp, sub_1, sum_1
# tensor => full_default
triton_per_fused__softmax_add_eq_lift_fresh_lt_native_dropout_3 = async_compile.triton('triton_per_fused__softmax_add_eq_lift_fresh_lt_native_dropout_3', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*i1', 4: '*fp32', 5: '*i1', 6: '*i1', 7: '*fp32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_eq_lift_fresh_lt_native_dropout_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_add_eq_lift_fresh_lt_native_dropout_3(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = -3.4028234663852886e+38
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tl.load(in_ptr2 + load_seed_offset)
    tmp16 = r2 + (128*x3)
    tmp17 = tl.rand(tmp15, (tmp16).to(tl.uint32))
    tmp18 = 0.1
    tmp19 = tmp17 > tmp18
    tmp20 = tmp10 / tmp14
    tmp21 = tmp2 == tmp3
    tmp22 = tmp2 < tmp3
    tmp23 = tmp19.to(tl.float32)
    tmp24 = tmp23 * tmp20
    tmp25 = 1.1111111111111112
    tmp26 = tmp24 * tmp25
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp19, rmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp20, rmask)
    tl.store(out_ptr5 + (r2 + (128*x3)), tmp21, rmask)
    tl.store(out_ptr6 + (r2 + (128*x3)), tmp22, rmask)
    tl.store(out_ptr7 + (r2 + (128*x3)), tmp26, rmask)
''')
