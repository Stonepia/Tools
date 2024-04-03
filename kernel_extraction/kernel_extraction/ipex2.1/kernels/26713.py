

# Original file: ./opacus_cifar10__32_inference_72.12/opacus_cifar10__32_inference_72.12.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/uf/cufgwbzaouly6tfcjwhdzkuypadm2ujlgmt7lb6ckymemiljcfzf.py
# Source Nodes: [iadd, l__self___relu], Original ATen: [aten.add, aten.relu]
# iadd => add
# l__self___relu => relu
triton_poi_fused_add_relu_0 = async_compile.triton('triton_poi_fused_add_relu_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_relu_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_relu_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
''')
