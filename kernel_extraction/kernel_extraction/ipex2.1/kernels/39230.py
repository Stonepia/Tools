

# Original file: ./maml__24_forward_72.5/maml__24_forward_72.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/sf/csfsdruwekzqv4ps3q5aip57rrnxn76yzv45k5bhkvolke4gxoqh.py
# Source Nodes: [batch_norm_1, relu_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# batch_norm_1 => add_4, add_5, add_6, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
# relu_1 => relu_1
triton_per_fused__native_batch_norm_legit_functional_relu_4 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_relu_4', '''
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
    size_hints=[64, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_relu_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_relu_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 180
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex % 36
    r2 = (rindex // 36)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (2304*r2)), rmask & xmask, other=0.0)
    tmp27 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 180, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = 180.0
    tmp19 = tmp17 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = 1.005586592178771
    tmp24 = tmp19 * tmp23
    tmp25 = 0.1
    tmp26 = tmp24 * tmp25
    tmp28 = 0.9
    tmp29 = tmp27 * tmp28
    tmp30 = tmp26 + tmp29
    tmp31 = tmp11 * tmp25
    tmp33 = tmp32 * tmp28
    tmp34 = tmp31 + tmp33
    tl.store(out_ptr2 + (x0), tmp22, xmask)
    tl.store(out_ptr3 + (x0), tmp30, xmask)
    tl.store(out_ptr4 + (x0), tmp34, xmask)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
''')
