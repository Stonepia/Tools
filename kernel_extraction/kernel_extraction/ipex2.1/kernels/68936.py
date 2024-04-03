

# Original file: ./doctr_det_predictor___60.0/doctr_det_predictor___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/vo/cvo5uuizc2tuqp5b2t2nuz42vaftjmezargt7llnuxakozl75vw6.py
# Source Nodes: [l__self___prob_head_3, l__self___prob_head_5, l__self___prob_head_6, sigmoid], Original ATen: [aten.convolution, aten.relu, aten.sigmoid]
# l__self___prob_head_3 => convolution_default_189
# l__self___prob_head_5 => relu_58
# l__self___prob_head_6 => convolution_63
# sigmoid => sigmoid
triton_poi_fused_convolution_relu_sigmoid_15 = async_compile.triton('triton_poi_fused_convolution_relu_sigmoid_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_sigmoid_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_relu_sigmoid_15(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (0)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tl.store(in_out_ptr0 + (x0), tmp3, None)
    tl.store(out_ptr0 + (x0), tmp4, None)
''')
