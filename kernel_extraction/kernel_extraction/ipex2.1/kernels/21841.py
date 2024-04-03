

# Original file: ./hrnet_w18___60.0/hrnet_w18___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/d3/cd32ixd3xwkopncx7zm6ca7ygeh3dlywergjfxifuaf3zartsstg.py
# Source Nodes: [add_2, add_3, l__self___stage3_0_fuse_layers_0_1_1, l__self___stage3_0_fuse_layers_0_1_2, l__self___stage3_0_fuse_layers_0_2_1, l__self___stage3_0_fuse_layers_0_2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten._to_copy, aten._unsafe_index, aten.add]
# add_2 => add_152
# add_3 => add_157
# l__self___stage3_0_fuse_layers_0_1_1 => add_149, mul_185, mul_186, sub_60
# l__self___stage3_0_fuse_layers_0_1_2 => _unsafe_index_1, convert_element_type_256
# l__self___stage3_0_fuse_layers_0_2_1 => add_154, mul_192, mul_193, sub_61
# l__self___stage3_0_fuse_layers_0_2_2 => _unsafe_index_2, convert_element_type_266
triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*bf16', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7225344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 1008) % 56
    x1 = (xindex // 18) % 56
    x0 = xindex % 18
    x3 = (xindex // 56448)
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp1 = x2
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.int32)
    tmp10 = x1
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp3
    tmp13 = tmp12 + tmp5
    tmp14 = tmp13 * tmp7
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr1 + (x0 + (18*tmp15) + (504*tmp9) + (14112*x3)), None).to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 - tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp0 + tmp26
    tmp28 = 0.25
    tmp29 = tmp6 * tmp28
    tmp30 = tmp29.to(tl.int32)
    tmp31 = tmp13 * tmp28
    tmp32 = tmp31.to(tl.int32)
    tmp33 = tl.load(in_ptr6 + (x0 + (18*tmp32) + (252*tmp30) + (3528*x3)), None).to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp36 = tmp34 - tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp27 + tmp43
    tl.store(out_ptr0 + (x4), tmp44, None)
''')
