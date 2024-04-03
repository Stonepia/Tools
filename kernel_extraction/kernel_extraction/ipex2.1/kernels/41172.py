

# Original file: ./fastNLP_Bert__21_inference_61.1/fastNLP_Bert__21_inference_61.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/zt/czt6h4nimlj2zn7ganduyxe7lxty2t5fgb23wlvpmyhlah33dbs2.py
# Source Nodes: [add_4, erf, mul_1, mul_2, truediv_1], Original ATen: [aten.add, aten.div, aten.erf, aten.mul]
# add_4 => add_8
# erf => erf
# mul_1 => mul_5
# mul_2 => mul_6
# truediv_1 => div_2
triton_poi_fused_add_div_erf_mul_5 = async_compile.triton('triton_poi_fused_add_div_erf_mul_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_erf_mul_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_erf_mul_5(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1459200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 1.4142135623730951
    tmp4 = tmp0 / tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')
