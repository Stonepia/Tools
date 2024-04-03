

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/of/cofmiyobr452kj4rfsmzw6ob35sbqscl5eauhd55mwjcoiziwx6k.py
# Source Nodes: [l__self___model_0, l__self___model_1], Original ATen: [aten._to_copy, aten.reflection_pad2d]
# l__self___model_0 => reflection_pad2d
# l__self___model_1 => convert_element_type
triton_poi_fused__to_copy_reflection_pad2d_0 = async_compile.triton('triton_poi_fused__to_copy_reflection_pad2d_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_reflection_pad2d_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_reflection_pad2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205932
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 786)
    x1 = (xindex // 3) % 262
    x0 = xindex % 3
    x5 = xindex
    tmp0 = (-3) + x2
    tmp1 = tl.abs(tmp0)
    tmp2 = tl.full([1], 255, tl.int32)
    tmp3 = tmp2 - tmp1
    tmp4 = tl.abs(tmp3)
    tmp5 = tmp2 - tmp4
    tmp6 = (-3) + x1
    tmp7 = tl.abs(tmp6)
    tmp8 = tmp2 - tmp7
    tmp9 = tl.abs(tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.load(in_ptr0 + (tmp10 + (256*tmp5) + (65536*x0)), xmask)
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x5), tmp12, xmask)
''')
