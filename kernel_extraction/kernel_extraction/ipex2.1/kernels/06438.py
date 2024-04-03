

# Original file: ./maml__29_forward_88.14/maml__29_forward_88.14_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/ir/cirrjsg4h4tlwzdp2rgcflcmczk3uza5qvt4gpaih7fwv2jxpois.py
# Source Nodes: [batch_norm_2, relu_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# batch_norm_2 => add_11, add_8, convert_element_type_11, convert_element_type_12, mul_14, mul_20, rsqrt_2, sub_4, var_mean_2
# relu_2 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 4*ks0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.0
    tmp9 = tmp7 - tmp8
    tmp10 = tmp5 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp4 * tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')
