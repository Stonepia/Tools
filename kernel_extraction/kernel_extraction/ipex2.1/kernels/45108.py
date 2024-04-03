

# Original file: ./eca_halonext26ts___60.0/eca_halonext26ts___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/xm/cxm6hktdlexmuvefdtqw2h7qqwd42hwlzqtznxdefpy2n5n7mjmf.py
# Source Nodes: [matmul_8], Original ATen: [aten.bmm]
# matmul_8 => bmm_4
triton_poi_fused_bmm_42 = async_compile.triton('triton_poi_fused_bmm_42', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_42', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bmm_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x1 = (xindex // 16384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*(x1 % 8)) + (1024*((((8*(x1 // 8)) + (x1 % 8)) // 8) % 8)) + (8192*((((8*(x1 // 8)) + (64*x0) + (x1 % 8)) // 8192) % 128)) + ((((8*(x1 // 8)) + (64*x0) + (x1 % 8)) // 64) % 128)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')
