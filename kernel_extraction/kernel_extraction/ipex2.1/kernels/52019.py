

# Original file: ./maml__22_forward_66.3/maml__22_forward_66.3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/sb/csbbxtjsq4in27s5wpsswcegqryl25jjsvnov6lbwglc4rdcbs7y.py
# Source Nodes: [batch_norm, batch_norm_1, batch_norm_2, batch_norm_3, conv2d, conv2d_1, conv2d_2, conv2d_3, mul_13, relu, relu_1, relu_2, relu_3, sub_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution, aten.mul, aten.relu, aten.sub]
# batch_norm => add, add_3, mul_18, mul_24, rsqrt, sub_18, var_mean
# batch_norm_1 => add_4, add_7, mul_25, mul_31, rsqrt_1, sub_19, var_mean_1
# batch_norm_2 => add_11, add_8, mul_32, mul_38, rsqrt_2, sub_20, var_mean_2
# batch_norm_3 => add_13, add_14, mul_40, mul_41, mul_42, mul_43, mul_44, var_mean_3
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# conv2d_3 => convolution_3
# mul_13 => mul_13
# relu => relu
# relu_1 => relu_1
# relu_2 => relu_2
# relu_3 => relu_3
# sub_13 => sub_13
triton_per_fused__native_batch_norm_legit_functional_convolution_mul_relu_sub_11 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_convolution_mul_relu_sub_11', '''
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
    size_hints=[64, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_convolution_mul_relu_sub_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_convolution_mul_relu_sub_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 75
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 0.4
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp6 = tmp5 + tmp0
    tmp7 = triton_helpers.maximum(0, tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tl.full([XBLOCK, 1], 75, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 / tmp16
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = 0.1
    tmp25 = tmp17 * tmp24
    tmp27 = 0.9
    tmp28 = tmp26 * tmp27
    tmp29 = tmp25 + tmp28
    tmp30 = 75.0
    tmp31 = tmp23 / tmp30
    tmp32 = 1.0135135135135136
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33 * tmp24
    tmp36 = tmp35 * tmp27
    tmp37 = tmp34 + tmp36
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr3 + (x0), tmp29, xmask)
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tl.store(out_ptr2 + (x0), tmp23, xmask)
''')
