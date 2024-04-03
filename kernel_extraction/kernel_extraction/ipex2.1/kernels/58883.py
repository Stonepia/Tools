

# Original file: ./nvidia_deeprecommender___60.0/nvidia_deeprecommender___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/xi/cxi55po633qeeqkgqgukjx7cz4r53od6hj62f4relwswnhstm3rm.py
# Source Nodes: [selu_2], Original ATen: [aten.elu]
# selu_2 => convert_element_type_11, convert_element_type_12, expm1_2, gt_2, mul_6, mul_7, mul_8, where_2
triton_poi_fused_elu_2 = async_compile.triton('triton_poi_fused_elu_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_elu_2(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp4 = 1.0507009873554805
    tmp5 = tmp1 * tmp4
    tmp6 = 1.0
    tmp7 = tmp1 * tmp6
    tmp8 = libdevice.expm1(tmp7)
    tmp9 = 1.7580993408473766
    tmp10 = tmp8 * tmp9
    tmp11 = tl.where(tmp3, tmp5, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp12, None)
''')
