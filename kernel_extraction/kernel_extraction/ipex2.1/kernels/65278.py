

# Original file: ./pytorch_unet___60.0/pytorch_unet___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/3b/c3bnkni3gvniz2mi4l64fg5ebdl37e55c5xyte65jhllp4uk2lwd.py
# Source Nodes: [l__mod___up1_up, pad], Original ATen: [aten._to_copy, aten.add, aten.arange, aten.constant_pad_nd, aten.mul, aten.rsub, aten.sub]
# l__mod___up1_up => add_21, add_24, convert_element_type_32, convert_element_type_35, convert_element_type_37, iota_1, mul_31, mul_33, mul_38, mul_39, sub_12, sub_13
# pad => constant_pad_nd
triton_poi_fused__to_copy_add_arange_constant_pad_nd_mul_rsub_sub_7 = async_compile.triton('triton_poi_fused__to_copy_add_arange_constant_pad_nd_mul_rsub_sub_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[512, 16384], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_constant_pad_nd_mul_rsub_sub_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_arange_constant_pad_nd_mul_rsub_sub_7(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 9520
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 119
    x2 = (xindex // 119)
    y0 = yindex
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1, 1], 118, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x1 + (118*x2) + (9440*y0)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.broadcast_to(x1, [XBLOCK, YBLOCK])
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = 0.0
    tmp9 = tmp7 + tmp8
    tmp10 = 0.49572649572649574
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 - tmp13
    tmp15 = tmp6 - tmp14
    tmp16 = tmp3 * tmp15
    tmp17 = tl.load(in_ptr1 + (x1 + (118*x2) + (9440*y0)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 * tmp14
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.where(tmp2, tmp20, 0.0)
    tl.store(out_ptr0 + (y0 + (1024*x3)), tmp21, xmask & ymask)
''')
