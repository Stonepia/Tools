

# Original file: ./nfnet_l0___60.0/nfnet_l0___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/fr/cfrhjibl4nxttoxwexklpyl6biiwtmptrblrgzn5oellrj4nbxt6.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
# getattr_getattr_l__mod___stages___3_____0___downsample_pool => avg_pool2d_2
triton_poi_fused_avg_pool2d_28 = async_compile.triton('triton_poi_fused_avg_pool2d_28', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15925248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1536
    x1 = (xindex // 1536) % 9
    x2 = (xindex // 13824)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*x1) + (55296*x2)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1536 + x0 + (3072*x1) + (55296*x2)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (27648 + x0 + (3072*x1) + (55296*x2)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (29184 + x0 + (3072*x1) + (55296*x2)), None).to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''')
