

# Original file: ./vision_maskrcnn__23_inference_63.3/vision_maskrcnn__23_inference_63.3_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/lk/clkzuhawvf566xcueqg33lxj3656vp45qegvxxdufhyva7bmlufu.py
# Source Nodes: [copy_, new_full], Original ATen: [aten.copy, aten.new_full, aten.select_scatter, aten.slice_scatter]
# copy_ => copy, select_scatter, slice_scatter
# new_full => full
triton_poi_fused_copy_new_full_select_scatter_slice_scatter_0 = async_compile.triton('triton_poi_fused_copy_new_full_select_scatter_slice_scatter_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_new_full_select_scatter_slice_scatter_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_new_full_select_scatter_slice_scatter_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2918400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1216
    x1 = (xindex // 1216)
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tmp0 == tmp0
    tmp2 = x0
    tmp3 = tl.full([1], 1199, tl.int64)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (1199*x1)), tmp4, other=0.0).to(tl.float32)
    tmp6 = tl.where(tmp4, tmp5, 0.0)
    tmp7 = 0.0
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tl.where(tmp1, tmp8, tmp7)
    tl.store(out_ptr0 + (x2), tmp9, None)
''')
