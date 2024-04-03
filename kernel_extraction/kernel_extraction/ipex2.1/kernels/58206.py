

# Original file: ./vision_maskrcnn__24_inference_64.4/vision_maskrcnn__24_inference_64.4_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/rc/crcqstf52awrjdfo5p4gygoxy3h34gpbiyrrglmxoll7zrlnjtgv.py
# Source Nodes: [add_105, add_99, getattr_l__self___backbone_body_layer4___1___relu_2, getattr_l__self___backbone_body_layer4___2___relu_2, iadd_14, iadd_15, l__self___backbone_fpn_inner_blocks_3_0, mul_149, mul_158], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.relu]
# add_105 => add_120
# add_99 => add_113
# getattr_l__self___backbone_body_layer4___1___relu_2 => relu_45
# getattr_l__self___backbone_body_layer4___2___relu_2 => relu_48
# iadd_14 => add_114
# iadd_15 => add_121
# l__self___backbone_fpn_inner_blocks_3_0 => convert_element_type_106
# mul_149 => mul_149
# mul_158 => mul_158
triton_poi_fused__to_copy_add_mul_relu_24 = async_compile.triton('triton_poi_fused__to_copy_add_mul_relu_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_relu_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1945600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''')
