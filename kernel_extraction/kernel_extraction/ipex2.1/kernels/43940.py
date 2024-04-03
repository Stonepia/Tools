

# Original file: ./yolov3___60.0/yolov3___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/wy/cwyrlzvgbjdwl6dfv7vlmryriqwusfbko5x3eywsvbfpnbjnsnyp.py
# Source Nodes: [l__self___module_list_103_activation, l__self___module_list_103_batch_norm2d, l__self___module_list_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten._to_copy, aten._unsafe_index, aten.leaky_relu]
# l__self___module_list_103_activation => gt_66, mul_275, where_66
# l__self___module_list_103_batch_norm2d => add_160, mul_273, mul_274, sub_66
# l__self___module_list_104 => _unsafe_index_1, convert_element_type_418
triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_leaky_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_leaky_relu_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_leaky_relu_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_leaky_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3072
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 64)
    x2 = xindex % 64
    y0 = yindex % 128
    y1 = (yindex // 128)
    x4 = xindex
    y5 = yindex
    tmp17 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp0 = x3
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x2
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.load(in_ptr0 + (y0 + (128*tmp14) + (4096*tmp8) + (98304*y1)), xmask).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 - tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22 + tmp17
    tmp24 = tmp23 > tmp4
    tmp25 = 0.1
    tmp26 = tmp23 * tmp25
    tmp27 = tl.where(tmp24, tmp23, tmp26)
    tmp28 = tmp27.to(tl.float32)
    tl.store(out_ptr1 + (y0 + (384*x4) + (1179648*y1)), tmp28, xmask)
''')
