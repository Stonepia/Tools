

# Original file: ./tacotron2__22_inference_62.2/tacotron2__22_inference_62.2_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/kz/ckzc6xcrunvlblu5ham42472ln4xhqaz3bd6wzchpe62yvqtjpis.py
# Source Nodes: [dropout, relu], Original ATen: [aten.native_dropout, aten.relu]
# dropout => gt, mul, mul_1
# relu => relu
triton_poi_fused_native_dropout_relu_2 = async_compile.triton('triton_poi_fused_native_dropout_relu_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_relu_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_native_dropout_relu_2(in_out_ptr0, in_ptr0, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14057472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp7 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.5
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tmp6 * tmp8
    tmp10 = 2.0
    tmp11 = tmp9 * tmp10
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''')
