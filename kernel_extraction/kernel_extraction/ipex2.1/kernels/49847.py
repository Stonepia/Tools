

# Original file: ./MobileBertForMaskedLM__0_forward_280.0/MobileBertForMaskedLM__0_forward_280.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/cb/ccb2b6hmqo6lou7uoh4ktz4frpvczrfjyliburgyh4e4m4b44dqv.py
# Source Nodes: [add_346, add_347, add_361, add_362, l__self___cls_predictions_transform_dense, l__self___mobilebert_encoder_layer_22_output_bottleneck_dropout, l__self___mobilebert_encoder_layer_23_output_bottleneck_dropout, mul_185, mul_193], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mul, aten.view]
# add_346 => add_346
# add_347 => add_347
# add_361 => add_361
# add_362 => add_362
# l__self___cls_predictions_transform_dense => convert_element_type_987, view_962
# l__self___mobilebert_encoder_layer_22_output_bottleneck_dropout => clone_115
# l__self___mobilebert_encoder_layer_23_output_bottleneck_dropout => clone_120
# mul_185 => mul_231
# mul_193 => mul_241
triton_poi_fused__to_copy_add_clone_mul_view_26 = async_compile.triton('triton_poi_fused__to_copy_add_clone_mul_view_26', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*bf16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_mul_view_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_clone_mul_view_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp1 + tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp15, None)
''')
