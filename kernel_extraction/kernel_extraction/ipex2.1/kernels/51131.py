

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/kj/ckj45kk7j5tlv3tynjnytnl75fzn6fxk73irp4suelaifwggm7je.py
# Source Nodes: [setitem_4], Original ATen: [aten.slice_scatter]
# setitem_4 => slice_scatter_15, slice_scatter_16
triton_poi_fused_slice_scatter_9 = async_compile.triton('triton_poi_fused_slice_scatter_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25214976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 513) % 4096
    x2 = (xindex // 2101248)
    x3 = xindex % 2101248
    x0 = xindex % 513
    x4 = xindex
    tmp8 = tl.load(in_ptr1 + (x0 + (513*(x1 % 256)) + (131328*x2)), None).to(tl.float32)
    tmp24 = tl.load(in_out_ptr0 + (x4), None).to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x3 + (131328*x2)), tmp2, other=0.0).to(tl.float32)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tmp5 = (x1 // 256)
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = tmp5 == tmp6
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp5 >= tmp9
    tmp11 = x0
    tmp12 = tmp11 < tmp1
    tmp13 = tmp12 & tmp10
    tmp14 = (((-131584) + x0 + (513*(x1 % 256)) + (262656*(x1 // 256)) + (3939840*x2)) // 512) % 513
    tmp15 = tl.full([1], 512, tl.int64)
    tmp16 = tmp14 < tmp15
    tmp17 = tmp16 & tmp13
    tmp18 = tl.load(in_ptr2 + ((512*((((-131584) + x0 + (513*(x1 % 256)) + (262656*(x1 // 256)) + (3939840*x2)) // 512) % 513)) + (262144*((((-131584) + x0 + (513*(x1 % 256)) + (262656*(x1 // 256)) + (3939840*x2)) // 262656) % 180)) + ((x0 + (513*(x1 % 256))) % 512)), tmp17, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tl.where(tmp13, tmp19, 0.0)
    tmp21 = tl.load(in_out_ptr0 + (x4), tmp10, other=0.0).to(tl.float32)
    tmp22 = tl.where(tmp12, tmp20, tmp21)
    tmp23 = tl.where(tmp10, tmp22, 0.0)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tl.where(tmp7, tmp8, tmp25)
    tmp27 = tl.where(tmp2, tmp4, tmp26)
    tl.store(in_out_ptr0 + (x4), tmp27, None)
''')
