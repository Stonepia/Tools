

# Original file: ./hrnet_w18___60.0/hrnet_w18___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/h4/ch4hmq2r6d32dmqsateojbveylw47mjjn7t5p2a2n6yeqh2jtiab.py
# Source Nodes: [add_28, l__self___stage4_0_fuse_act, l__self___stage4_0_fuse_layers_0_3_1, l__self___stage4_0_fuse_layers_0_3_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten._to_copy, aten._unsafe_index, aten.add, aten.relu]
# add_28 => add_528
# l__self___stage4_0_fuse_act => relu_180
# l__self___stage4_0_fuse_layers_0_3_1 => add_525, mul_646, mul_647, sub_195
# l__self___stage4_0_fuse_layers_0_3_2 => _unsafe_index_15, convert_element_type_880
triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 18
    y1 = (yindex // 18)
    x3 = (xindex // 56)
    x2 = xindex % 56
    tmp0 = tl.load(in_out_ptr0 + (y0 + (18*x4) + (56448*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = x3
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.125
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.int32)
    tmp10 = x2
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp3
    tmp13 = tmp12 + tmp5
    tmp14 = tmp13 * tmp7
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (y0 + (18*tmp15) + (126*tmp9) + (882*y1)), xmask & ymask).to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 - tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp0 + tmp26
    tmp28 = triton_helpers.maximum(0, tmp27)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (18*x4) + (56448*y1)), tmp28, xmask & ymask)
''')
