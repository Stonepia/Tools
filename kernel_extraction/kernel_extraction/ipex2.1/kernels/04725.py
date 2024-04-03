

# Original file: ./DebertaForMaskedLM__0_forward_133.0/DebertaForMaskedLM__0_forward_133.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/bk/cbkufsrichcl34fuwtdmpfavl4pov5ovoow2o3cttvjyqon3itly.py
# Source Nodes: [add_2, matmul, sqrt_1, truediv_1], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.div, aten.sqrt]
# add_2 => add_3
# matmul => clone, convert_element_type_4
# sqrt_1 => full_default_2
# truediv_1 => div_1
triton_poi_fused__to_copy_add_clone_div_sqrt_3 = async_compile.triton('triton_poi_fused__to_copy_add_clone_div_sqrt_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_div_sqrt_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_clone_div_sqrt_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*x2) + (2304*x1) + (1179648*x3)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = 8.0
    tmp5 = tmp3 / tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp6, None)
''')
