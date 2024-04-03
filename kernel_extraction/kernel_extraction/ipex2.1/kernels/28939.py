

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/bl/cblg42qoejgmgvr4db32gjhg4ycvhk4mt5ihm5nibbfyfzfa2xen.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___mlp_act], Original ATen: [aten.silu]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___mlp_act => convert_element_type_189, convert_element_type_190, mul_161, sigmoid_28
triton_poi_fused_silu_55 = async_compile.triton('triton_poi_fused_silu_55', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_55', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_silu_55(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''')
