

# Original file: ./tacotron2__25_inference_65.5/tacotron2__25_inference_65.5_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/p2/cp2qh3wixuu2nsgpyjixovzcajwb7ume6rmzrmwsaxrj7a7qzfbp.py
# Source Nodes: [cat_3429, l_____stack0_____self___attention_rnn_856], Original ATen: [aten.add, aten.cat, aten.mul, aten.sigmoid, aten.tanh]
# cat_3429 => cat_3426
# l_____stack0_____self___attention_rnn_856 => add_5993, mul_5136, mul_5137, mul_5138, sigmoid_5136, sigmoid_5137, sigmoid_5138, tanh_4280, tanh_4281
triton_poi_fused_add_cat_mul_sigmoid_tanh_879 = async_compile.triton('triton_poi_fused_add_cat_mul_sigmoid_tanh_879', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_sigmoid_tanh_879', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_cat_mul_sigmoid_tanh_879(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + x0 + (4096*x1)), None).to(tl.float32)
    tmp2 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (x0 + (4096*x1)), None).to(tl.float32)
    tmp6 = tl.load(in_ptr0 + (2048 + x0 + (4096*x1)), None).to(tl.float32)
    tmp10 = tl.load(in_ptr0 + (3072 + x0 + (4096*x1)), None).to(tl.float32)
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tmp1 * tmp2
    tmp5 = tl.sigmoid(tmp4)
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = tmp3 + tmp8
    tmp11 = tl.sigmoid(tmp10)
    tmp12 = libdevice.tanh(tmp9)
    tmp13 = tmp11 * tmp12
    tl.store(in_out_ptr0 + (x2), tmp9, None)
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x0 + (1536*x1)), tmp13, None)
''')