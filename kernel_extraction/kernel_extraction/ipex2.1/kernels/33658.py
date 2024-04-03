

# Original file: ./functorch_dp_cifar10___60.0/functorch_dp_cifar10___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/6j/c6jhdfndvqww56r5yhzwzg6phknsivtqmdttrgz5d3pfdo3uqw5j.py
# Source Nodes: [getattr_l__self___layer4___1___bn2, getattr_l__self___layer4___1___relu_1, iadd_7, l__self___avgpool], Original ATen: [aten._adaptive_avg_pool2d, aten.add, aten.native_group_norm, aten.relu]
# getattr_l__self___layer4___1___bn2 => add_46, convert_element_type_98, mul_39
# getattr_l__self___layer4___1___relu_1 => relu_16
# iadd_7 => add_47
# l__self___avgpool => _adaptive_avg_pool2d
triton_poi_fused__adaptive_avg_pool2d_add_native_group_norm_relu_18 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_add_native_group_norm_relu_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_add_native_group_norm_relu_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_add_native_group_norm_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((32*x1) + (x0 // 16)), None)
    tmp4 = tl.load(in_ptr2 + ((32*x1) + (x0 // 16)), None)
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = triton_helpers.maximum(0, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''')
