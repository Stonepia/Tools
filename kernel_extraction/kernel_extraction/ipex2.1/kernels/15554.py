

# Original file: ./tacotron2__25_inference_65.5/tacotron2__25_inference_65.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/qi/cqixdpkzzra73s5252bryzwmvgns4f65nlvejkgdnepa2qxlhgpr.py
# Source Nodes: [cat_6853, l_____stack0_____self___attention_rnn], Original ATen: [aten.add, aten.cat, aten.mul, aten.sigmoid, aten.tanh]
# cat_6853 => cat_2
# l_____stack0_____self___attention_rnn => add_1, mul, mul_1, mul_2, sigmoid, sigmoid_1, sigmoid_2, tanh, tanh_1
triton_poi_fused_add_cat_mul_sigmoid_tanh_6 = async_compile.triton('triton_poi_fused_add_cat_mul_sigmoid_tanh_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_sigmoid_tanh_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_cat_mul_sigmoid_tanh_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (3072 + x0 + (4096*x1)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (1024 + x0 + (4096*x1)), None).to(tl.float32)
    tmp6 = tl.load(in_ptr0 + (x0 + (4096*x1)), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (2048 + x0 + (4096*x1)), None).to(tl.float32)
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
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x0 + (1536*x1)), tmp13, None)
''')
