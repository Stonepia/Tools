

# Original file: ./speech_transformer__25_inference_65.5/speech_transformer__25_inference_65.5_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/ck/cck2mybqkj6nlkkm3gt5xcaelundly74vyzerhhpqe6j6wcaeohq.py
# Source Nodes: [add, gt], Original ATen: [aten.add, aten.gt]
# add => add
# gt => gt
triton_poi_fused_add_gt_1 = async_compile.triton('triton_poi_fused_add_gt_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gt_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_gt_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 22
    x2 = (xindex // 484)
    x1 = (xindex // 22) % 22
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (22*x2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2.to(tl.int8).to(tl.uint8)
    tmp4 = x0 + ((-1)*x1)
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 1, tl.int16).to(tl.int8).to(tl.uint8)
    tmp8 = tl.full([1], 0, tl.int16).to(tl.int8).to(tl.uint8)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp3 + tmp9
    tmp11 = tmp10 > tmp8
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')
