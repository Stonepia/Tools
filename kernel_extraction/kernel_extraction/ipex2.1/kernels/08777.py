

# Original file: ./jx_nest_base___60.0/jx_nest_base___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/zi/czitnvtdojjsmfbnfeli7stuxufs7hxumadodd457uncealk3h4h.py
# Source Nodes: [getattr_l__self___levels___1___pool_conv], Original ATen: [aten._to_copy]
# getattr_l__self___levels___1___pool_conv => convert_element_type_31
triton_poi_fused__to_copy_11 = async_compile.triton('triton_poi_fused__to_copy_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 56
    x2 = (xindex // 7168) % 56
    x3 = (xindex // 401408)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*(x1 % 14)) + (1792*(x2 % 14)) + (25088*(x1 // 14)) + (100352*(x2 // 14)) + (401408*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*(x1 % 14)) + (1792*(x2 % 14)) + (25088*(x1 // 14)) + (100352*(x2 // 14)) + (401408*x3)), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp4, None)
''')
