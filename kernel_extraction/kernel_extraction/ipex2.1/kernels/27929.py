

# Original file: ./maml__23_forward_69.4/maml__23_forward_69.4.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/22/c22p2rius7rvz7x7zleilyclktnf3xzvorh5bsdlpo4tsxn3dv26.py
# Source Nodes: [batch_norm, batch_norm_1, batch_norm_2, conv2d, conv2d_1, conv2d_2, relu, relu_1, relu_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution, aten.relu]
# batch_norm => add, add_3, mul, mul_6, rsqrt, sub, var_mean
# batch_norm_1 => add_4, add_7, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
# batch_norm_2 => add_10, add_9, mul_15, mul_16, mul_17, mul_18, mul_19, var_mean_2
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# relu => relu
# relu_1 => relu_1
# relu_2 => relu_2
triton_per_fused__native_batch_norm_legit_functional_convolution_relu_5 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_convolution_relu_5', '''
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
    size_hints=[64, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_convolution_relu_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_convolution_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 300
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex % 4
    r2 = (rindex // 4)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0) + (256*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.full([1], 300, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 0.1
    tmp21 = tmp13 * tmp20
    tmp23 = 0.9
    tmp24 = tmp22 * tmp23
    tmp25 = tmp21 + tmp24
    tmp26 = 300.0
    tmp27 = tmp19 / tmp26
    tmp28 = 1.0033444816053512
    tmp29 = tmp27 * tmp28
    tmp30 = tmp29 * tmp20
    tmp32 = tmp31 * tmp23
    tmp33 = tmp30 + tmp32
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp33, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp19, xmask)
''')
