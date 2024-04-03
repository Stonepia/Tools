

# Original file: ./tacotron2__25_inference_65.5/tacotron2__25_inference_65.5_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/c6/cc6wq7wsad3zvpsfiswnkidduziumle5qef37aqu3tzqetsow7tr.py
# Source Nodes: [l_____stack0_____self___decoder_rnn], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.tanh]
# l_____stack0_____self___decoder_rnn => add_6, mul_3, mul_4, mul_5, sigmoid_3, sigmoid_4, sigmoid_5, tanh_3, tanh_4
triton_poi_fused_add_mul_sigmoid_tanh_14 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_tanh_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_tanh_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_sigmoid_tanh_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (3072 + x0 + (4096*x1)), None)
    tmp2 = tl.load(in_ptr0 + (1024 + x0 + (4096*x1)), None)
    tmp6 = tl.load(in_ptr0 + (x0 + (4096*x1)), None)
    tmp8 = tl.load(in_ptr0 + (2048 + x0 + (4096*x1)), None)
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 * tmp4
    tmp7 = tl.sigmoid(tmp6)
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = tmp7 * tmp9
    tmp11 = tmp5 + tmp10
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = tmp1 * tmp12
    tl.store(out_ptr0 + (x0 + (1536*x1)), tmp13, None)
''')
