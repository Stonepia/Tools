

# Original file: ./MegatronBertForCausalLM__0_backward_351.1/MegatronBertForCausalLM__0_backward_351.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/k2/ck2z2gkaatzrhk4fuddmia36w4rwrhsfmbx4j2hfhkuioyuk7r55.py
# Source Nodes: [cross_entropy], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.nll_loss_forward]
# cross_entropy => full_default_3
triton_poi_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_21 = async_compile.triton('triton_poi_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_21(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29753344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')
