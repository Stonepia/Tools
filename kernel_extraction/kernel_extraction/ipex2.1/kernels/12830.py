

# Original file: ./hf_Reformer__25_inference_65.5/hf_Reformer__25_inference_65.5_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/du/cdu5izgk4qboym6o3r7rmhh7wpbiwsznxvkxj2lsir26cmimkmrf.py
# Source Nodes: [cat_71, cat_73], Original ATen: [aten.cat]
# cat_71 => cat_10
# cat_73 => cat_8
triton_poi_fused_cat_25 = async_compile.triton('triton_poi_fused_cat_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i64', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3096576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 4032
    x2 = (xindex // 258048)
    x0 = xindex % 64
    x3 = xindex % 258048
    tmp0 = tl.load(in_ptr0 + (x1 + (4096*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1 + (4096*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.full([1], 4096, tl.int64)
    tmp2 = tmp0 % tmp1
    tmp3 = tmp2 + tmp1
    tmp4 = tl.where(((tmp2 != 0) & ((tmp2 < 0) != (tmp1 < 0))), tmp3, tmp2)
    tmp5 = tl.where(tmp4 < 0, tmp4 + 4096, tmp4)
    # tl.device_assert((0 <= tmp5) & (tmp5 < 4096), "index out of bounds: 0 <= tmp5 < 4096")
    tmp6 = tl.load(in_ptr1 + (x0 + (64*x2) + (768*tmp5)), None).to(tl.float32)
    tmp8 = 64.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp6 * tmp13
    tmp15 = 8.0
    tmp16 = tmp14 / tmp15
    tmp17 = tl.load(in_ptr3 + (x0 + (64*x2) + (768*tmp5)), None).to(tl.float32)
    tl.store(out_ptr0 + (x3 + (262144*x2)), tmp16, None)
    tl.store(out_ptr1 + (x3 + (262144*x2)), tmp17, None)
''')
