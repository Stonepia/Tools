

# Original file: ./visformer_small___60.0/visformer_small___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/bz/cbzc7enttzqu3gfkauxgb2dasx26k2shatgnniujd5hdsavja3so.py
# Source Nodes: [matmul_8], Original ATen: [aten.clone]
# matmul_8 => clone_49
triton_poi_fused_clone_25 = async_compile.triton('triton_poi_fused_clone_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 49
    x2 = (xindex // 6272) % 6
    x3 = (xindex // 37632)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x2) + (2304*x1) + (112896*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')
