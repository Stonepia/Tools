

# Original file: ./maml__21_backward_64.2/maml__21_backward_64.2_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ed/cedqqlmu427nguemqnehi2234ltdrsr7cxd56dmgqijasz54h4yx.py
# Source Nodes: [batch_norm, relu], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.relu]
# batch_norm => convert_element_type
# relu => relu
triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_11 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_11', '''
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
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 845
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex % 169
    r2 = (rindex // 169)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (169*x0) + (10816*r2)), rmask & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (r1 + (169*x0) + (10816*r2)), rmask & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = triton_helpers.maximum(0, tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp11 = tmp1 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp5.to(tl.float32)
    tl.store(out_ptr2 + (x0), tmp18, xmask)
    tl.store(out_ptr3 + (x0), tmp19, xmask)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
''')