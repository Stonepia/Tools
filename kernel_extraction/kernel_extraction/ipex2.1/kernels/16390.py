

# Original file: ./tacotron2__25_inference_65.5/tacotron2__25_inference_65.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/jl/cjl4g5cyfnr5foumgms6w4vwse2s5m6xt27i2sknwl7h2rspu7ti.py
# Source Nodes: [cat_3575], Original ATen: [aten.cat]
# cat_3575 => cat_3280
triton_poi_fused_cat_842 = async_compile.triton('triton_poi_fused_cat_842', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_842', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_842(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (13434880 + x2), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (768*x1)), tmp0, None)
''')
