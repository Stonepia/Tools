

# Original file: ./hf_Reformer__25_inference_65.5/hf_Reformer__25_inference_65.5_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ck/cckyamybg3efsutaxti7wqpwzgn2igvl3uav4orpd4uuavdcwi5a.py
# Source Nodes: [cat_69], Original ATen: [aten.cat]
# cat_69 => cat_12
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (4032 + x0 + (4096*x1)), xmask)
    tmp1 = tl.full([1], 4096, tl.int64)
    tmp2 = tmp0 % tmp1
    tmp3 = tmp2 + tmp1
    tmp4 = tl.where(((tmp2 != 0) & ((tmp2 < 0) != (tmp1 < 0))), tmp3, tmp2)
    tl.store(out_ptr0 + (x0 + (4096*x1)), tmp4, xmask)
''')
