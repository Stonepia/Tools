

# Original file: ./vision_maskrcnn__24_inference_64.4/vision_maskrcnn__24_inference_64.4.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/fy/cfy6qjdodmgsn7twt2dfwrhuvuxwowwrcr42dx5n4shjhvjmwis4.py
# Source Nodes: [add_99, getattr_l__self___backbone_body_layer4___1___relu_2, getattr_l__self___backbone_body_layer4___2___conv1, iadd_14, mul_149], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.relu]
# add_99 => add_113
# getattr_l__self___backbone_body_layer4___1___relu_2 => relu_45
# getattr_l__self___backbone_body_layer4___2___conv1 => convert_element_type_100
# iadd_14 => add_114
# mul_149 => mul_149
triton_poi_fused__to_copy_add_mul_relu_23 = async_compile.triton('triton_poi_fused__to_copy_add_mul_relu_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_relu_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1945600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), None)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp9, None)
''')
