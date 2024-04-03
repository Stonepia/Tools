

# Original file: ./eca_halonext26ts___60.0/eca_halonext26ts___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/7m/c7mv6txettg7zmxr27tdujci3stxxlg42zqyyvdhc425jc2esb5w.py
# Source Nodes: [matmul, reshape_3], Original ATen: [aten.clone]
# matmul => clone_2
# reshape_3 => clone_4
triton_poi_fused_clone_13 = async_compile.triton('triton_poi_fused_clone_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_13(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 64
    x2 = (xindex // 1024) % 4
    x3 = (xindex // 4096)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*(((8*(x2 % 2)) + (x1 % 8)) % 16)) + (2048*((((8*(x2 % 2)) + (16*(x1 // 8)) + (128*(x2 // 2)) + (x1 % 8)) // 16) % 16)) + (32768*((((8*(x2 % 2)) + (16*(x1 // 8)) + (128*(x2 // 2)) + (256*x0) + (4096*x3) + (x1 % 8)) // 32768) % 128)) + ((((8*(x2 % 2)) + (16*(x1 // 8)) + (128*(x2 // 2)) + (256*x0) + (4096*x3) + (x1 % 8)) // 256) % 128)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
    tl.store(out_ptr1 + (x4), tmp0, None)
''')
