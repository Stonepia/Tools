

# Original file: ./maml__23_forward_69.4/maml__23_forward_69.4_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/pt/cptm6fbwi6jx3odq3ber57l37z52yghxdh7o64zbk3i7gqiw4jzh.py
# Source Nodes: [batch_norm, batch_norm_1, batch_norm_2, conv2d, conv2d_1, conv2d_2, relu, relu_1, relu_2], Original ATen: [aten._native_batch_norm_legit_functional, aten._to_copy, aten.convolution, aten.relu]
# batch_norm => add, add_3, convert_element_type_3, convert_element_type_4, mul, mul_6, rsqrt, sub, var_mean
# batch_norm_1 => add_4, add_7, convert_element_type_7, convert_element_type_8, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
# batch_norm_2 => add_10, add_9, convert_element_type_11, mul_15, mul_16, mul_17, mul_18, mul_19, var_mean_2
# conv2d => convert_element_type, convert_element_type_1, convert_element_type_2, convolution
# conv2d_1 => convert_element_type_5, convert_element_type_6, convolution_1
# conv2d_2 => convert_element_type_10, convert_element_type_9, convolution_2
# relu => relu
# relu_1 => relu_1
# relu_2 => relu_2
triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_9 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_9', '''
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
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    x0 = xindex
    r1 = rindex % 4
    r2 = (rindex // 4)
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (r1 + (4*x0) + (256*r2)), rmask & xmask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2 + tmp1
    tmp4 = triton_helpers.maximum(0, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 300, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = 0.1
    tmp23 = tmp15 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 300.0
    tmp29 = tmp21 / tmp28
    tmp30 = 1.0033444816053512
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31 * tmp22
    tmp34 = tmp33 * tmp25
    tmp35 = tmp32 + tmp34
    tl.store(out_ptr0 + (x0), tmp1, xmask)
    tl.store(out_ptr3 + (x0), tmp27, xmask)
    tl.store(out_ptr4 + (x0), tmp35, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tl.store(out_ptr2 + (x0), tmp21, xmask)
''')
