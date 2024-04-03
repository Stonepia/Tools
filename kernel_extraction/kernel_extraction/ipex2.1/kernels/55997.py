

# Original file: ./cait_m36_384___60.0/cait_m36_384___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/jw/cjwmn3icvc5irhbps4ahs6hopfryqxlxuwudpxqumf2yawshk2qh.py
# Source Nodes: [add_71, add_72, cat_3, cat_4, getattr_l__self___blocks___35___attn_proj_drop, getattr_l__self___blocks___35___mlp_drop2, mul_106, mul_107], Original ATen: [aten.add, aten.cat, aten.clone, aten.mul]
# add_71 => add_320
# add_72 => add_324
# cat_3 => cat_2
# cat_4 => cat_1
# getattr_l__self___blocks___35___attn_proj_drop => clone_285
# getattr_l__self___blocks___35___mlp_drop2 => clone_288
# mul_106 => mul_353
# mul_107 => mul_359
triton_poi_fused_add_cat_clone_mul_13 = async_compile.triton('triton_poi_fused_add_cat_clone_mul_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_clone_mul_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_cat_clone_mul_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp5 + tmp9
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(out_ptr1 + (x2), tmp10, None)
    tl.store(out_ptr2 + (x2), tmp10, None)
''')
