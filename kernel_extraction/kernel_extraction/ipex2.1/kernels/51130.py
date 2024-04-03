

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/lg/clg224aloebt6pfq66xoasvuy3zbpb5ver5dzp7tmmvkurhdw46m.py
# Source Nodes: [setitem_4], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_4 => copy_4, slice_scatter_14
triton_poi_fused_copy_slice_scatter_8 = async_compile.triton('triton_poi_fused_copy_slice_scatter_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1575936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 513
    x3 = (xindex // 513)
    x4 = xindex
    x1 = (xindex // 513) % 256
    x2 = (xindex // 131328)
    x5 = xindex % 131328
    tmp7 = tl.load(in_ptr1 + (x4), xmask).to(tl.float32)
    tmp24 = tl.load(in_ptr3 + (x5 + (2101248*x2)), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 257, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (257*x3)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = tmp5 == tmp5
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1], 256, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = (((-131584) + x0 + (513*x1) + (3939840*x2)) // 512) % 513
    tmp15 = tl.full([1], 512, tl.int64)
    tmp16 = tmp14 < tmp15
    tmp17 = tmp16 & tmp13
    tmp18 = tl.load(in_ptr2 + ((512*((((-131584) + x5 + (3939840*x2)) // 512) % 513)) + (262144*((((-131584) + x5 + (3939840*x2)) // 262656) % 180)) + (x5 % 512)), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tl.where(tmp13, tmp19, 0.0)
    tmp21 = tl.load(in_ptr3 + (x5 + (2101248*x2)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.where(tmp12, tmp20, tmp21)
    tmp23 = tl.where(tmp10, tmp22, 0.0)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tl.where(tmp6, tmp7, tmp25)
    tmp27 = tl.where(tmp2, tmp4, tmp26)
    tl.store(out_ptr0 + (x4), tmp27, xmask)
''')
