

# Original file: ./swin_base_patch4_window7_224___60.0/swin_base_patch4_window7_224___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/f4/cf4tsiekh7oh5wr2mh5h3rqb64fd2yed54hha4kgojr6grfyv3zr.py
# Source Nodes: [flatten_1], Original ATen: [aten.clone]
# flatten_1 => clone_46
triton_poi_fused_clone_27 = async_compile.triton('triton_poi_fused_clone_27', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256) % 28
    x2 = (xindex // 7168) % 28
    x3 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*(x1 % 7)) + (1792*(x2 % 7)) + (12544*(x1 // 7)) + (50176*(x2 // 7)) + (200704*x3)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0 + (256*(((25 + x1) % 28) % 7)) + (1792*(((25 + x2) % 28) % 7)) + (12544*(((25 + x1) % 28) // 7)) + (50176*(((25 + x2) % 28) // 7)) + (200704*x3)), None).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x4), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x4), tmp8, None)
''')
