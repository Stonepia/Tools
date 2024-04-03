

# Original file: ./maml__23_forward_69.4/maml__23_forward_69.4_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/xw/cxwzakeqweq552vdmdivmirpqyxdhl3opgpg7635zwdqoole44ch.py
# Source Nodes: [batch_norm, conv2d, relu], Original ATen: [aten._native_batch_norm_legit_functional, aten._to_copy, aten.convolution, aten.relu]
# batch_norm => add_1, add_2, convert_element_type, convert_element_type_2, convert_element_type_3, mul_1, mul_2, mul_3, mul_4, mul_5, var_mean
# conv2d => convolution
# relu => relu
triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_1 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_1', '''
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
    size_hints=[64, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*bf16', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp29 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 0.1
    tmp17 = tmp13 * tmp16
    tmp19 = 0.9
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp17 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 12675.0
    tmp25 = tmp14 / tmp24
    tmp26 = 1.0000789016884961
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27 * tmp16
    tmp30 = tmp29 * tmp19
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp28 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr2 + (x0), tmp23, xmask)
    tl.store(out_ptr3 + (x0), tmp33, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')
