

# Original file: ./functorch_dp_cifar10___60.0/functorch_dp_cifar10___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/4x/c4x2si6lgatfhsnsblr7ggtbz2zpzqveork6ctngvm4pruczffen.py
# Source Nodes: [getattr_l__mod___layer2___0___bn2, getattr_l__mod___layer2___0___downsample_1, getattr_l__mod___layer2___0___relu_1, iadd_2], Original ATen: [aten.add, aten.native_group_norm, aten.relu]
# getattr_l__mod___layer2___0___bn2 => add_15, convert_element_type_25, mul_13
# getattr_l__mod___layer2___0___downsample_1 => add_17, convert_element_type_29, mul_15
# getattr_l__mod___layer2___0___relu_1 => relu_6
# iadd_2 => add_18
triton_poi_fused_add_native_group_norm_relu_9 = async_compile.triton('triton_poi_fused_add_native_group_norm_relu_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_group_norm_relu_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_native_group_norm_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = (xindex // 2048)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((32*x2) + (x0 // 4)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + ((32*x2) + (x0 // 4)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr5 + (x3), None).to(tl.float32)
    tmp20 = tl.load(in_ptr6 + ((32*x2) + (x0 // 4)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + ((32*x2) + (x0 // 4)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp19 - tmp20
    tmp23 = tmp22 / tmp5
    tmp24 = tmp23 + tmp7
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp21 * tmp25
    tmp27 = tmp26 * tmp12
    tmp28 = tmp27 + tmp15
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp17 + tmp29
    tmp31 = triton_helpers.maximum(0, tmp30)
    tl.store(out_ptr0 + (x3), tmp31, None)
''')
