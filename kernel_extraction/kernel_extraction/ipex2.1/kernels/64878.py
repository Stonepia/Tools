

# Original file: ./demucs__23_inference_63.3/demucs__23_inference_63.3_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/kk/ckkjqf5xgu6feahzwol337fji4ldogyqz3v2p35ukjo2w3qy3iht.py
# Source Nodes: [add, add_1, l__self___decoder_0_0, l__self___decoder_0_1, l__self___decoder_0_2, l__self___decoder_0_3], Original ATen: [aten.add, aten.convolution, aten.glu, aten.relu]
# add => add
# add_1 => add_1
# l__self___decoder_0_0 => convolution
# l__self___decoder_0_1 => glu
# l__self___decoder_0_2 => convolution_1
# l__self___decoder_0_3 => relu
triton_poi_fused_add_convolution_glu_relu_2 = async_compile.triton('triton_poi_fused_add_convolution_glu_relu_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_glu_relu_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_convolution_glu_relu_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2981888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 364) % 1024
    x0 = xindex % 364
    x4 = (xindex // 364)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (4 + x0 + (372*x4)), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp5 = tmp3 + tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''')
