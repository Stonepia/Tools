

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/j2/cj23ul6flewfellmb6z64qwycwbcg3q7cswcnm5effuu7f5yhqcm.py
# Source Nodes: [setitem_3], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_3 => copy_3, slice_scatter_11, slice_scatter_12
triton_poi_fused_copy_slice_scatter_6 = async_compile.triton('triton_poi_fused_copy_slice_scatter_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1575936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 513) % 256
    x0 = xindex % 513
    x2 = (xindex // 131328)
    x3 = xindex % 131328
    x4 = xindex
    tmp41 = tl.load(in_ptr1 + (x3 + (2101248*x2)), xmask).to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = (((-256) + x0 + (513*x1) + (3939840*x2)) // 512) % 513
    tmp10 = tl.full([1], 512, tl.int64)
    tmp11 = tmp9 < tmp10
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + ((262144*((((-256) + x0 + (513*x1) + (3939840*x2)) // 262656) % 180)) + (((-256) + x0 + (513*x1) + (3939840*x2)) % 262656)), tmp12 & xmask, other=0.0).to(tl.float32)
    tmp14 = tl.where(tmp12, tmp13, 0.0)
    tmp15 = tl.where(tmp8, tmp14, 0.0)
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp16 >= tmp1
    tmp18 = tmp17 & tmp2
    tmp19 = tmp6 & tmp18
    tmp20 = (((-131584) + x0 + (513*x1) + (3939840*x2)) // 512) % 513
    tmp21 = tmp20 < tmp10
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + ((512*((((-131584) + x3 + (3939840*x2)) // 512) % 513)) + (262144*((((-131584) + x3 + (3939840*x2)) // 262656) % 180)) + (x3 % 512)), tmp22 & xmask, other=0.0).to(tl.float32)
    tmp24 = tl.where(tmp22, tmp23, 0.0)
    tmp25 = tl.where(tmp19, tmp24, 0.0)
    tmp26 = tl.load(in_ptr1 + (x3 + (2101248*x2)), tmp18 & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp6, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp27, 0.0)
    tmp29 = tl.load(in_ptr1 + (x3 + (2101248*x2)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp30 = tl.where(tmp17, tmp28, tmp29)
    tmp31 = tl.where(tmp7, tmp15, tmp30)
    tmp32 = tl.where(tmp2, tmp31, 0.0)
    tmp33 = tmp6 & tmp17
    tmp34 = tmp21 & tmp33
    tmp35 = tl.load(in_ptr0 + ((512*((((-131584) + x3 + (3939840*x2)) // 512) % 513)) + (262144*((((-131584) + x3 + (3939840*x2)) // 262656) % 180)) + (x3 % 512)), tmp34 & xmask, other=0.0).to(tl.float32)
    tmp36 = tl.where(tmp34, tmp35, 0.0)
    tmp37 = tl.where(tmp33, tmp36, 0.0)
    tmp38 = tl.load(in_ptr1 + (x3 + (2101248*x2)), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp39 = tl.where(tmp6, tmp37, tmp38)
    tmp40 = tl.where(tmp17, tmp39, 0.0)
    tmp42 = tl.where(tmp17, tmp40, tmp41)
    tmp43 = tl.where(tmp2, tmp32, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, xmask)
''')
