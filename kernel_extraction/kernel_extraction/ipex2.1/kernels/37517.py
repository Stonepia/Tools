

# Original file: ./BlenderbotForCausalLM__91_forward_278.25/BlenderbotForCausalLM__91_forward_278.25_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/k7/ck7idw7xivpuxtnffnpvhcxqnvzui35sowohs3k2liy3xjp6xb7y.py
# Source Nodes: [l__self___self_attn_out_proj], Original ATen: [aten.view]
# l__self___self_attn_out_proj => view_16
triton_poi_fused_view_6 = async_compile.triton('triton_poi_fused_view_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2560
    x1 = (xindex // 2560)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((80*(x1 % 128)) + (10240*(x0 // 80)) + (327680*(x1 // 128)) + (x0 % 80)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')