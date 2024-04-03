

# Original file: ./hf_Albert___60.0/hf_Albert___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/cp/ccpt3nt24s77tlpokz47a3utwyplp7lptep2fn4x42x7s6a47yww.py
# Source Nodes: [l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query, l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value], Original ATen: [aten.addmm]
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_key => _linear_pointwise_default_72
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_query => _linear_pointwise_default_73
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_value => _linear_pointwise_default_71
triton_poi_fused_addmm_1 = async_compile.triton('triton_poi_fused_addmm_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_addmm_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
    tl.store(out_ptr2 + (x0), tmp0, None)
''')
