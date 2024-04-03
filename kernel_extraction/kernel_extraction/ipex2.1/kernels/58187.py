

# Original file: ./vision_maskrcnn__24_inference_64.4/vision_maskrcnn__24_inference_64.4_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/4r/c4r4ghtt3irftl6vmr46cmlpgfmggk7gadh4rcwudsf6siov4iwh.py
# Source Nodes: [add_7, add_9, getattr_l__self___backbone_body_layer1___0___relu_2, getattr_l__self___backbone_body_layer1___1___conv1, iadd, mul_11, mul_14], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.relu]
# add_7 => add_7
# add_9 => add_9
# getattr_l__self___backbone_body_layer1___0___relu_2 => relu_3
# getattr_l__self___backbone_body_layer1___1___conv1 => convert_element_type_10
# iadd => add_10
# mul_11 => mul_11
# mul_14 => mul_14
triton_poi_fused__to_copy_add_mul_relu_5 = async_compile.triton('triton_poi_fused__to_copy_add_mul_relu_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_relu_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15564800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp5 + tmp11
    tmp13 = triton_helpers.maximum(0, tmp12)
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp14, None)
''')