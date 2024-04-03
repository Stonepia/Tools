

# Original file: ./functorch_dp_cifar10___60.0/functorch_dp_cifar10___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/jp/cjp6cdtxdwm6nwmxx7d32hpbc6fnwxi4chxatkaaltm2mjcip2ue.py
# Source Nodes: [getattr_l__self___layer4___0___bn2, getattr_l__self___layer4___0___downsample_1, getattr_l__self___layer4___0___relu_1, iadd_6], Original ATen: [aten.add, aten.native_group_norm, aten.relu]
# getattr_l__self___layer4___0___bn2 => add_39, convert_element_type_83, mul_33
# getattr_l__self___layer4___0___downsample_1 => add_41, convert_element_type_88, mul_35
# getattr_l__self___layer4___0___relu_1 => relu_14
# iadd_6 => add_42
triton_poi_fused_add_native_group_norm_relu_17 = async_compile.triton('triton_poi_fused_add_native_group_norm_relu_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_group_norm_relu_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_native_group_norm_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp18 = tl.load(in_ptr6 + ((32*x1) + (x0 // 16)), None)
    tmp20 = tl.load(in_ptr7 + ((32*x1) + (x0 // 16)), None)
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
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 - tmp18
    tmp21 = tmp20 / tmp5
    tmp22 = tmp21 + tmp7
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp19 * tmp23
    tmp25 = tmp24 * tmp11
    tmp26 = tmp25 + tmp13
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp15 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (x2), tmp29, None)
''')
