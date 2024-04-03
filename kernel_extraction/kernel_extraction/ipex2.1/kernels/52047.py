

# Original file: ./maml__22_forward_66.3/maml__22_forward_66.3_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/p3/cp3bszz5efheowatoigz2jybh3cary3ps4duibavtyk56vpcog2b.py
# Source Nodes: [batch_norm, conv2d, relu], Original ATen: [aten._native_batch_norm_legit_functional, aten._to_copy, aten.convolution, aten.relu]
# batch_norm => convert_element_type_3, var_mean
# conv2d => convert_element_type, convert_element_type_1, convert_element_type_2, convolution
# relu => relu
triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_5 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6338
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    tmp16_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (6338*x1)
        tmp1 = tl.full([1, 1], 12675, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((169*x0) + (10816*(((r2 + (6338*x1)) // 169) % 75)) + ((r2 + (6338*x1)) % 169)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = triton_helpers.maximum(0, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.where(tmp2, tmp7, 0)
        tmp9 = 0.0
        tmp10 = tl.where(tmp2, tmp9, 0)
        tmp11 = 1.0
        tmp12 = tl.where(tmp2, tmp11, 0)
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp15 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_combine(
            tmp16_mean, tmp16_m2, tmp16_weight,
            tmp13, tmp14, tmp15
        )
        tmp16_mean = tl.where(rmask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(rmask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(rmask & xmask, tmp16_weight_next, tmp16_weight)
    tmp16_tmp, tmp17_tmp, tmp18_tmp = triton_helpers.welford(
        tmp16_mean, tmp16_m2, tmp16_weight, 1
    )
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
    tl.store(out_ptr2 + (x3), tmp18, xmask)
''')
