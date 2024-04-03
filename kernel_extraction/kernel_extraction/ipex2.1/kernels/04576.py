

# Original file: ./tacotron2__28_inference_68.8/tacotron2__28_inference_68.8_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/fl/cfl3gfjq6xz2un4jznj3fykvjmf7vp6lst6cx4dh2mdk2gtpbjdo.py
# Source Nodes: [masked_fill_, masked_fill__1], Original ATen: [aten.masked_fill]
# masked_fill_ => clone, full_default, where
# masked_fill__1 => clone_1, where_1
triton_poi_fused_masked_fill_0 = async_compile.triton('triton_poi_fused_masked_fill_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr0', 'out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_masked_fill_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_masked_fill_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4387840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 80)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask)
    tmp5 = tl.load(in_ptr2 + (x2), xmask)
    tmp1 = tmp0 == 0
    tmp3 = 0.0
    tmp4 = tl.where(tmp1, tmp3, tmp2)
    tmp6 = tl.where(tmp1, tmp3, tmp5)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr1 + (x2), tmp6, xmask)
''')
