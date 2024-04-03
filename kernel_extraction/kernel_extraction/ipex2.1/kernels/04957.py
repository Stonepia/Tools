

# Original file: ./swin_base_patch4_window7_224___60.0/swin_base_patch4_window7_224___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/dq/cdqycavy4imlq6qr7xa733qeep7v5edqnnmltphsaus2sbezqwwu.py
# Source Nodes: [flatten], Original ATen: [aten.clone]
# flatten => clone_23
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

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128) % 56
    x2 = (xindex // 7168) % 56
    x3 = (xindex // 401408)
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*(x1 % 7)) + (896*(x2 % 7)) + (6272*(x1 // 7)) + (50176*(x2 // 7)) + (401408*x3)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0 + (128*(((53 + x1) % 56) % 7)) + (896*(((53 + x2) % 56) % 7)) + (6272*(((53 + x1) % 56) // 7)) + (50176*(((53 + x2) % 56) // 7)) + (401408*x3)), None).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x4), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x4), tmp8, None)
''')