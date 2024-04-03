

# Original file: ./XGLMForCausalLM__21_inference_61.1/XGLMForCausalLM__21_inference_61.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/65/c65kigifoxvtm5kccoenmudtsxelunrybrf7iamqkcrlw4xmahj2.py
# Source Nodes: [iadd], Original ATen: [aten.add]
# iadd => add
triton_poi_fused_add_1 = async_compile.triton('triton_poi_fused_add_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')
