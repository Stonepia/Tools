

# Original file: ./MobileBertForMaskedLM__0_forward_352.0/MobileBertForMaskedLM__0_forward_352.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/qo/cqocfbbd3an75pz7rbmkwmhkidakxp3d4jav5zdhtaqtevmb6uhl.py
# Source Nodes: [add_17, add_31, add_32, l__self___mobilebert_encoder_layer_1_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_2_bottleneck_input_dense, mul_17, mul_9], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mul, aten.view]
# add_17 => add_17
# add_31 => add_31
# add_32 => add_32
# l__self___mobilebert_encoder_layer_1_output_bottleneck_dropout => clone_10
# l__self___mobilebert_encoder_layer_2_bottleneck_input_dense => convert_element_type_85, view_82
# mul_17 => mul_21
# mul_9 => mul_11
triton_poi_fused__to_copy_add_clone_mul_view_23 = async_compile.triton('triton_poi_fused__to_copy_add_clone_mul_view_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_mul_view_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_clone_mul_view_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp1 + tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp11, None)
    tl.store(out_ptr1 + (x2), tmp12, None)
''')
