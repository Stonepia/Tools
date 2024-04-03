

# Original file: ./tf_mixnet_l___60.0/tf_mixnet_l___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/qy/cqygkj5bjsmpdhf6eyjmm3bblzaawkxfgupkk4thxtd4bfjhjiz7.py
# Source Nodes: [pad_11], Original ATen: [aten.constant_pad_nd]
# pad_11 => constant_pad_nd_11
triton_poi_fused_constant_pad_nd_86 = async_compile.triton('triton_poi_fused_constant_pad_nd_86', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_86', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_86(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6912000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3600) % 15
    x1 = (xindex // 240) % 15
    x0 = xindex % 240
    x3 = (xindex // 54000)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 14, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (960*x1) + (13440*x2) + (188160*x3)), tmp5, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.where(tmp5, tmp9, 0.0)
    tl.store(out_ptr0 + (x4), tmp10, None)
''')
