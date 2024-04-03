

# Original file: ./hf_T5_generate__85_inference_125.65/hf_T5_generate__85_inference_125.65_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ri/crimcot2yedjden6pze5mfp7g5xymbdji5u7idzlnlhfjh7wynxn.py
# Source Nodes: [l__self___decoder_block_0_layer__1__dense_relu_dense_act], Original ATen: [aten.relu]
# l__self___decoder_block_0_layer__1__dense_relu_dense_act => relu
triton_poi_fused_relu_10 = async_compile.triton('triton_poi_fused_relu_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_relu_10(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp1, None)
''')
