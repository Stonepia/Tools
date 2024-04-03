

# Original file: ./maml__24_forward_72.5/maml__24_forward_72.5_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/hk/chk7tqm6odwrq7zf2s753ynaqxnncu4duxahcz24grtu6wvslr6g.py
# Source Nodes: [batch_norm_1, relu_1], Original ATen: [aten._native_batch_norm_legit_functional, aten._to_copy, aten.relu]
# batch_norm_1 => add_4, add_5, add_6, convert_element_type_4, convert_element_type_6, convert_element_type_7, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
# relu_1 => relu_1
triton_per_fused__native_batch_norm_legit_functional__to_copy_relu_4 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional__to_copy_relu_4', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional__to_copy_relu_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__to_copy_relu_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (36*x0) + (2304*r2)), rmask & xmask, other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp35 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 180, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 180.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = 1.005586592178771
    tmp25 = tmp20 * tmp24
    tmp26 = 0.1
    tmp27 = tmp25 * tmp26
    tmp29 = 0.9
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp27 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp12 * tmp26
    tmp36 = tmp35 * tmp29
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp34 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr2 + (x0), tmp23, xmask)
    tl.store(out_ptr3 + (x0), tmp33, xmask)
    tl.store(out_ptr4 + (x0), tmp39, xmask)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
''')
