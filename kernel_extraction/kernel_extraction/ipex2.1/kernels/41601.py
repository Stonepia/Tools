

# Original file: ./maml__21_forward_62.1/maml__21_forward_62.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/ry/cry3txovdrgcqpd5jgtfdokos5s45rzcit5f5msybfhvzbbrjkn7.py
# Source Nodes: [conv2d_3], Original ATen: [aten._to_copy, aten.convolution]
# conv2d_3 => convert_element_type_14, convolution_3
triton_poi_fused__to_copy_convolution_14 = async_compile.triton('triton_poi_fused__to_copy_convolution_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')
