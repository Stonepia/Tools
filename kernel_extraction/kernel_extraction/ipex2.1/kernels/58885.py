

# Original file: ./nvidia_deeprecommender___60.0/nvidia_deeprecommender___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ns/cnstnwe6zibf5ohchil2xoe6bzz427giojz7y7swwdos3nh5ozaf.py
# Source Nodes: [selu], Original ATen: [aten.elu]
# selu => expm1, gt, mul, mul_1, mul_2, where
triton_poi_fused_elu_0 = async_compile.triton('triton_poi_fused_elu_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_elu_0(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0507009873554805
    tmp4 = tmp0 * tmp3
    tmp5 = 1.0
    tmp6 = tmp0 * tmp5
    tmp7 = libdevice.expm1(tmp6)
    tmp8 = 1.7580993408473766
    tmp9 = tmp7 * tmp8
    tmp10 = tl.where(tmp2, tmp4, tmp9)
    tl.store(in_out_ptr0 + (x0), tmp10, None)
''')
