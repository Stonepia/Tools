

# Original file: ./detectron2_fasterrcnn_r_101_c4__53_inference_93.33/detectron2_fasterrcnn_r_101_c4__53_inference_93.33_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/xp/cxpjpmgalxvoir3s4u6dwgot634ykgvomtm6w2aekdg6ulsyklyr.py
# Source Nodes: [getitem_3], Original ATen: [aten.index]
# getitem_3 => index_2
triton_poi_fused_index_0 = async_compile.triton('triton_poi_fused_index_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_index_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 106
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 2)
    x0 = xindex % 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 288, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 288)) | ~xmask, "index out of bounds: 0 <= tmp1 < 288")
    tmp2 = tl.load(in_ptr1 + (x0 + (2*tmp1)), xmask)
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''')