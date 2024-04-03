

# Original file: ./M2M100ForConditionalGeneration__96_forward_295.22/M2M100ForConditionalGeneration__96_forward_295.22_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/lz/clzstt4w2h3sfsayoxncvbxmajiplcmqmrs7enfwiy7daid43yb3.py
# Source Nodes: [dropout_4, l__self___activation_fn, l__self___fc2], Original ATen: [aten.clone, aten.relu, aten.threshold_backward, aten.view]
# dropout_4 => clone_8
# l__self___activation_fn => relu
# l__self___fc2 => view_36
triton_poi_fused_clone_relu_threshold_backward_view_13 = async_compile.triton('triton_poi_fused_clone_relu_threshold_backward_view_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_relu_threshold_backward_view_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_relu_threshold_backward_view_13(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tl.store(out_ptr0 + (x0), tmp1, None)
    tl.store(out_ptr1 + (x0), tmp3, None)
''')
