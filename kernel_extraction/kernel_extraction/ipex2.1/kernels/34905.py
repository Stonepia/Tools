

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/je/cjeyoy7p6ug6pngms6f2prp3jeu4fxzpbkxgvf3fzbsg7r2ylmtl.py
# Source Nodes: [setitem_22], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_22 => copy_22, slice_scatter_101
triton_poi_fused_copy_slice_scatter_28 = async_compile.triton('triton_poi_fused_copy_slice_scatter_28', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_28(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1045200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x2 = (xindex // 5226)
    x3 = xindex % 5226
    x1 = (xindex // 26) % 201
    x4 = xindex
    tmp42 = tl.load(in_ptr1 + (10634 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 2 + x2
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 202, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8 & tmp2
    tmp10 = tl.load(in_ptr0 + (26 + x3 + (5304*x2)), tmp9 & xmask, other=0.0).to(tl.float32)
    tmp11 = tl.where(tmp9, tmp10, 0.0)
    tmp12 = 1 + x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp6
    tmp15 = tmp13 & tmp14
    tmp16 = tmp15 & tmp9
    tmp17 = tmp2 & tmp16
    tmp18 = tl.load(in_ptr1 + (10634 + x3 + (5304*x2)), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tl.load(in_ptr1 + (10634 + x3 + (5304*x2)), tmp16 & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp2, tmp19, tmp20)
    tmp22 = tl.where(tmp16, tmp21, 0.0)
    tmp23 = tl.load(in_ptr1 + (10634 + x3 + (5304*x2)), tmp9 & xmask, other=0.0).to(tl.float32)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tl.where(tmp9, tmp24, 0.0)
    tmp26 = tl.load(in_ptr1 + (10634 + x3 + (5304*x2)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp8, tmp11, tmp27)
    tmp29 = tl.where(tmp2, tmp28, 0.0)
    tmp30 = tl.load(in_ptr0 + (26 + x3 + (5304*x2)), tmp8 & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp8, tmp30, 0.0)
    tmp32 = tmp15 & tmp8
    tmp33 = tmp2 & tmp32
    tmp34 = tl.load(in_ptr1 + (10634 + x3 + (5304*x2)), tmp33 & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, 0.0)
    tmp36 = tl.load(in_ptr1 + (10634 + x3 + (5304*x2)), tmp32 & xmask, other=0.0).to(tl.float32)
    tmp37 = tl.where(tmp2, tmp35, tmp36)
    tmp38 = tl.where(tmp32, tmp37, 0.0)
    tmp39 = tl.load(in_ptr1 + (10634 + x3 + (5304*x2)), tmp8 & xmask, other=0.0).to(tl.float32)
    tmp40 = tl.where(tmp15, tmp38, tmp39)
    tmp41 = tl.where(tmp8, tmp40, 0.0)
    tmp43 = tl.where(tmp8, tmp41, tmp42)
    tmp44 = tl.where(tmp8, tmp31, tmp43)
    tmp45 = tl.where(tmp2, tmp29, tmp44)
    tl.store(out_ptr0 + (x4), tmp45, xmask)
''')
