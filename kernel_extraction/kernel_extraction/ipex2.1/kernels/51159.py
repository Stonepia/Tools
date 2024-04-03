

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/id/cidoq5dgpnjzn4avtdz26zl5ypxdsniwdoumy6firn7gksrvarye.py
# Source Nodes: [bool_3, full_like_2, where_2], Original ATen: [aten._to_copy, aten.full_like, aten.where]
# bool_3 => convert_element_type_3
# full_like_2 => full_default_7
# where_2 => where_5
triton_poi_fused__to_copy_full_like_where_13 = async_compile.triton('triton_poi_fused__to_copy_full_like_where_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_full_like_where_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_full_like_where_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 257
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp9 = tl.load(in_ptr0 + (x1 + (513*y0)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp28 = tl.load(in_ptr3 + (y0 + (4096*x1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = (-255) + x1 + y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = (tmp5 != 0)
    tmp7 = tl.full([1, 1], 0, tl.int32)
    tmp8 = tmp7 == tmp7
    tmp10 = tl.full([1, 1], 1, tl.int64)
    tmp11 = tmp1 >= tmp10
    tmp12 = tl.broadcast_to(x1, [XBLOCK, YBLOCK])
    tmp13 = tl.full([1, 1], 256, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp14 & tmp11
    tmp16 = (((-131584) + x1 + (513*y0)) // 512) % 513
    tmp17 = tl.full([1, 1], 512, tl.int64)
    tmp18 = tmp16 < tmp17
    tmp19 = tmp18 & tmp15
    tmp20 = tl.load(in_ptr1 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 15)) + ((((-131584) + x1 + (513*y0)) // 512) % 513)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr2 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 15)) + ((x1 + (513*y0)) % 512)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.where(tmp19, tmp22, 0.0)
    tmp24 = tl.where(tmp15, tmp23, 0.0)
    tmp25 = tl.load(in_ptr3 + (y0 + (4096*x1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.where(tmp14, tmp24, tmp25)
    tmp27 = tl.where(tmp11, tmp26, 0.0)
    tmp29 = tl.where(tmp11, tmp27, tmp28)
    tmp30 = tl.where(tmp8, tmp9, tmp29)
    tmp31 = float("-inf")
    tmp32 = tl.where(tmp6, tmp31, tmp30)
    tl.store(out_ptr0 + (x1 + (257*y0)), tmp32, xmask & ymask)
''')
