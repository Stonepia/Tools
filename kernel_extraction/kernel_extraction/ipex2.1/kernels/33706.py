

# Original file: ./functorch_dp_cifar10___60.0/functorch_dp_cifar10___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/xu/cxukv2inx7of7oquymegxbawkzpbdgqk7mdiwpbwnabxkloanjbx.py
# Source Nodes: [getattr_l__mod___layer2___0___bn2, getattr_l__mod___layer2___0___downsample_1, getattr_l__mod___layer2___0___relu_1, iadd_2], Original ATen: [aten.add, aten.native_group_norm, aten.relu]
# getattr_l__mod___layer2___0___bn2 => add_15, mul_13
# getattr_l__mod___layer2___0___downsample_1 => add_17, mul_15
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

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_group_norm_relu_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_native_group_norm_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = (xindex // 2048)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + ((32*x2) + (x0 // 4)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((32*x2) + (x0 // 4)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp15 = tl.load(in_ptr6 + ((32*x2) + (x0 // 4)), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + ((32*x2) + (x0 // 4)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 64.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp22 = tmp21 * tmp10
    tmp23 = tmp22 + tmp12
    tmp24 = tmp13 + tmp23
    tmp25 = triton_helpers.maximum(0, tmp24)
    tl.store(out_ptr0 + (x3), tmp25, None)
''')
