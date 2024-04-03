

# Original file: ./DALLE2_pytorch__44_inference_84.24/DALLE2_pytorch__44_inference_84.24.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/or/corelcsts3kl6s4zit3fy5soacaprn6qajsxtjwvbskhshze37pf.py
# Source Nodes: [add, add_1, l__self___block1_act, l__self___block1_norm, mul], Original ATen: [aten.add, aten.mul, aten.native_group_norm, aten.silu]
# add => add_2
# add_1 => add_3
# l__self___block1_act => mul_4, sigmoid_1
# l__self___block1_norm => add_1, mul_2
# mul => mul_3
triton_poi_fused_add_mul_native_group_norm_silu_4 = async_compile.triton('triton_poi_fused_add_mul_native_group_norm_silu_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_group_norm_silu_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_native_group_norm_silu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + ((x0 // 16)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + ((x0 // 16)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (128 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 262144.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tmp17 = tmp13 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp19 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''')
