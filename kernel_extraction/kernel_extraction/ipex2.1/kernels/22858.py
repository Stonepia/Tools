

# Original file: ./mixnet_l___60.0/mixnet_l___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/5j/c5jqvnoenvgn5mglw3ysznjm35wf4ebjx2vcqmtfheilxr5plx6g.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_act1], Original ATen: [aten.silu]
# getattr_getattr_l__mod___blocks___3_____0___se_act1 => convert_element_type_97, convert_element_type_98, mul_87, sigmoid_18
triton_poi_fused_silu_48 = async_compile.triton('triton_poi_fused_silu_48', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_48', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_silu_48(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')
