

# Original file: ./vision_maskrcnn__24_inference_64.4/vision_maskrcnn__24_inference_64.4_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/m2/cm2esej6fbxbg4pzj5wqg2qkg6qy5yrrb36rw4t2vlh5r7trd34i.py
# Source Nodes: [add_85, getattr_l__self___backbone_body_layer3___5___relu_2, getattr_l__self___backbone_body_layer4___0___conv1, iadd_12, mul_128], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.relu]
# add_85 => add_97
# getattr_l__self___backbone_body_layer3___5___relu_2 => relu_39
# getattr_l__self___backbone_body_layer4___0___conv1 => convert_element_type_86
# iadd_12 => add_98
# mul_128 => mul_128
triton_poi_fused__to_copy_add_mul_relu_19 = async_compile.triton('triton_poi_fused__to_copy_add_mul_relu_19', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_relu_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3891200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2), None)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tmp8.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp9, None)
''')
