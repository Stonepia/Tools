

# Original file: ./XGLMForCausalLM__21_inference_61.1/XGLMForCausalLM__21_inference_61.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/fr/cfrvowur5kwivivsj7ojzpjwjdomg5wjbkvdvetxfau4bcgbo4an.py
# Source Nodes: [detach], Original ATen: [aten.detach]
# detach => alias
triton_poi_fused_detach_0 = async_compile.triton('triton_poi_fused_detach_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_detach_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_detach_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024)
    x0 = xindex % 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.where(tmp2 < 0, tmp2 + 2050, tmp2)
    # tl.device_assert((0 <= tmp3) & (tmp3 < 2050), "index out of bounds: 0 <= tmp3 < 2050")
    tmp4 = tl.load(in_ptr1 + (x0 + (1024*tmp3)), None)
    tl.store(out_ptr0 + (x2), tmp4, None)
''')
