

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/ao/caozvcxsizh4yk3dcl76y7cpf5knsfqoqromlvhxsihhnryup5re.py
# Source Nodes: [l__self___cell_9_conv_prev_1x1_path_1_avgpool], Original ATen: [aten.avg_pool2d]
# l__self___cell_9_conv_prev_1x1_path_1_avgpool => avg_pool2d_6
triton_poi_fused_avg_pool2d_52 = async_compile.triton('triton_poi_fused_avg_pool2d_52', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_52', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_52(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4181760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2160
    x1 = (xindex // 2160) % 11
    x2 = (xindex // 23760) % 11
    x3 = (xindex // 261360)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4320*x1) + (90720*x2) + (952560*x3)), xmask).to(tl.float32)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''')
