

# Original file: ./detectron2_maskrcnn_r_50_fpn__73_inference_113.53/detectron2_maskrcnn_r_50_fpn__73_inference_113.53_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ju/cjuihfzihj26sfwgbhejznqbhl5uz5w6j4gozkglv6afcoxrndln.py
# Source Nodes: [l__self___mask_head_0_activation], Original ATen: [aten.relu]
# l__self___mask_head_0_activation => fn_3
triton_poi_fused_relu_0 = async_compile.triton('triton_poi_fused_relu_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_relu_0(in_ptr0, out_ptr0, ks0, ks1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7936
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (ks0*ks1*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + (256*x2) + (256*ks0*ks1*y1)), tmp0, xmask & ymask)
''')
