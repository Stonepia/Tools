

# Original file: ./vision_maskrcnn__24_inference_64.4/vision_maskrcnn__24_inference_64.4_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/5h/c5h25wec2o7pxybu7762s65hclxbb3tynz3msuyzieaqz2aybovp.py
# Source Nodes: [add_61, add_67, getattr_l__self___backbone_body_layer3___1___relu_2, getattr_l__self___backbone_body_layer3___2___relu_2, getattr_l__self___backbone_body_layer3___3___conv1, iadd_8, iadd_9, mul_101, mul_92], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.relu]
# add_61 => add_69
# add_67 => add_76
# getattr_l__self___backbone_body_layer3___1___relu_2 => relu_27
# getattr_l__self___backbone_body_layer3___2___relu_2 => relu_30
# getattr_l__self___backbone_body_layer3___3___conv1 => convert_element_type_68
# iadd_8 => add_70
# iadd_9 => add_77
# mul_101 => mul_101
# mul_92 => mul_92
triton_poi_fused__to_copy_add_mul_relu_18 = async_compile.triton('triton_poi_fused__to_copy_add_mul_relu_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_relu_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3891200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = tmp5 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tmp17 = tmp16.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp16, None)
    tl.store(out_ptr0 + (x2), tmp17, None)
''')
