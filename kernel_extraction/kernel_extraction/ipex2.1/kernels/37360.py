

# Original file: ./BlenderbotForCausalLM__88_backward_285.29/BlenderbotForCausalLM__88_backward_285.29_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/6v/c6vak2dtd556bmz3zu6yvkkj5vkjzri67y3bhn7nfd6ydzijc5gj.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 80
    x1 = (xindex // 80) % 128
    x2 = (xindex // 10240) % 32
    x3 = (xindex // 327680)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (80*x2) + (2560*x1) + (327680*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')
