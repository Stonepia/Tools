

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/u2/cu2ibbvn7iimgznslempgjxjz7kgtabip47w24cvfcnzuilkixbm.py
# Source Nodes: [new_zeros_1, setitem_6, setitem_7], Original ATen: [aten.copy, aten.new_zeros, aten.select_scatter, aten.slice, aten.slice_scatter]
# new_zeros_1 => full_5
# setitem_6 => copy_6, slice_117, slice_scatter_22, slice_scatter_23, slice_scatter_24, slice_scatter_25
# setitem_7 => copy_7, select_scatter_2, slice_134, slice_scatter_26, slice_scatter_27
triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_11 = async_compile.triton('triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_new_zeros_select_scatter_slice_slice_scatter_11(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 513
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 256)
    y0 = yindex
    x1 = xindex % 256
    x3 = xindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 15, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = y0
    tmp4 = tl.full([1, 1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = ((3808256 + y0 + (513*x1)) // 512) % 513
    tmp7 = tl.full([1, 1], 512, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + ((256*((3808256 + y0 + (513*x1)) // 262656)) + (((3808256 + y0 + (513*x1)) // 512) % 513)), tmp9 & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + ((256*((3808256 + y0 + (513*x1)) // 262656)) + ((y0 + (513*x1)) % 512)), tmp9 & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tl.where(tmp9, tmp12, 0.0)
    tmp14 = tl.where(tmp5, tmp13, 0.0)
    tmp15 = tl.full([1, 1], 15, tl.int64)
    tmp16 = tmp15 < tmp15
    tmp17 = tl.broadcast_to(y0, [XBLOCK, YBLOCK])
    tmp18 = tmp17 >= tmp4
    tmp19 = tmp18 & tmp16
    tmp20 = ((3939584 + y0 + (513*x1)) // 512) % 513
    tmp21 = tmp20 < tmp7
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + ((256*(((3939584 + y0 + (513*x1)) // 262656) % 15)) + (((3939584 + y0 + (513*x1)) // 512) % 513)), tmp22 & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr1 + ((256*(((3939584 + y0 + (513*x1)) // 262656) % 15)) + ((3939584 + y0 + (513*x1)) % 512)), tmp22 & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.where(tmp22, tmp25, 0.0)
    tmp27 = tl.where(tmp19, tmp26, 0.0)
    tmp28 = 0.0
    tmp29 = tl.where(tmp18, tmp27, tmp28)
    tmp30 = tl.where(tmp16, tmp29, 0.0)
    tmp31 = tl.where(tmp16, tmp30, tmp28)
    tmp32 = tl.where(tmp5, tmp14, tmp31)
    tmp33 = tmp0 < tmp15
    tmp34 = tmp18 & tmp33
    tmp35 = (((-256) + y0 + (513*x1) + (262656*x2)) // 512) % 513
    tmp36 = tmp35 < tmp7
    tmp37 = tmp36 & tmp34
    tmp38 = tl.load(in_ptr0 + ((256*((((-256) + y0 + (513*x1) + (262656*x2)) // 262656) % 15)) + ((((-256) + y0 + (513*x1) + (262656*x2)) // 512) % 513)), tmp37 & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr1 + ((256*((((-256) + y0 + (513*x1) + (262656*x2)) // 262656) % 15)) + (((-256) + y0 + (513*x1) + (262656*x2)) % 512)), tmp37 & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tmp38 * tmp39
    tmp41 = tl.where(tmp37, tmp40, 0.0)
    tmp42 = tl.where(tmp34, tmp41, 0.0)
    tmp43 = tl.where(tmp18, tmp42, tmp28)
    tmp44 = tl.where(tmp33, tmp43, 0.0)
    tmp45 = tl.where(tmp33, tmp44, tmp28)
    tmp46 = tl.where(tmp2, tmp32, tmp45)
    tl.store(out_ptr0 + (x3 + (4096*y0)), tmp46, ymask)
''')
