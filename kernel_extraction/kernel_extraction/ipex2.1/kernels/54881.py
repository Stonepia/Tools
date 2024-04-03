

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/e5/ce5jdx55psssqozz3kt7zqtpxrkfvtosyq43zqlpkdqnv55ciigb.py
# Source Nodes: [setitem_4], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_4 => copy_4, slice_scatter_14
triton_poi_fused_copy_slice_scatter_10 = async_compile.triton('triton_poi_fused_copy_slice_scatter_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 513
    x4 = (xindex // 513)
    x5 = xindex
    x1 = (xindex // 513) % 256
    x6 = (xindex // 131328)
    x2 = (xindex // 131328) % 12
    x3 = (xindex // 1575936)
    x7 = xindex % 131328
    tmp7 = tl.load(in_ptr1 + (x5), None).to(tl.float32)
    tmp24 = tl.load(in_ptr3 + (x7 + (525312*x6)), None).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 257, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (257*x4)), tmp2, other=0.0).to(tl.float32)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = tmp5 == tmp5
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1], 256, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = (((-131584) + x0 + (513*x1) + (787968*x6)) // 512) % 513
    tmp15 = tl.full([1], 512, tl.int64)
    tmp16 = tmp14 < tmp15
    tmp17 = tmp16 & tmp13
    tmp18 = tl.load(in_ptr2 + ((512*((((-131584) + x7 + (787968*x2) + (9455616*x3)) // 512) % 513)) + (262144*((((-131584) + x7 + (787968*x2) + (9455616*x3)) // 262656) % 144)) + (x7 % 512)), tmp17, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tl.where(tmp13, tmp19, 0.0)
    tmp21 = tl.load(in_ptr3 + (x7 + (525312*x6)), tmp10, other=0.0).to(tl.float32)
    tmp22 = tl.where(tmp12, tmp20, tmp21)
    tmp23 = tl.where(tmp10, tmp22, 0.0)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tl.where(tmp6, tmp7, tmp25)
    tmp27 = tl.where(tmp2, tmp4, tmp26)
    tl.store(out_ptr0 + (x5), tmp27, None)
''')