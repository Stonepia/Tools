

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/fs/cfsebr2d2gml3lq3mifa4o7ztxpx4xl7fwpjwotf5dnbdvwiwgjy.py
# Source Nodes: [iadd, softmax], Original ATen: [aten._softmax, aten._to_copy, aten.add]
# iadd => add_2
# softmax => amax, clone_2, convert_element_type_5, exp, sub_4, sum_1
triton_per_fused__softmax__to_copy_add_15 = async_compile.triton('triton_per_fused__softmax__to_copy_add_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_15(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 49152
    XBLOCK: tl.constexpr = 1
    rnumel = 513
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x1 = (xindex // 12)
    r2 = rindex
    x0 = xindex % 12
    x3 = xindex
    tmp21 = tl.load(in_ptr0 + (r2 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 16)) + (2101248*x0)), rmask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr1 + (r2 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 16))), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 3840, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to(r2, [RBLOCK])
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = 4352 + ((-1)*r2) + ((-1)*x1)
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp7 <= tmp8
    tmp10 = 1.0
    tmp11 = 0.0
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = (tmp12 != 0)
    tmp14 = tl.load(in_ptr0 + (r2 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 16)) + (2101248*x0)), rmask & tmp6, other=0.0).to(tl.float32)
    tmp15 = float("-inf")
    tmp16 = tl.where(tmp13, tmp15, tmp14)
    tmp17 = tl.where(tmp6, tmp16, 0.0)
    tmp18 = tl.load(in_ptr0 + (r2 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 16)) + (2101248*x0)), rmask & tmp2, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp5, tmp17, tmp18)
    tmp20 = tl.where(tmp2, tmp19, 0.0)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = tl.load(in_ptr1 + (r2 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 16))), rmask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.where(tmp13, tmp15, tmp23)
    tmp25 = tl.where(tmp6, tmp24, 0.0)
    tmp26 = tl.load(in_ptr1 + (r2 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 16))), rmask & tmp2, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp5, tmp25, tmp26)
    tmp28 = tl.where(tmp2, tmp27, 0.0)
    tmp30 = tl.where(tmp2, tmp28, tmp29)
    tmp31 = tmp22 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, float("-inf"))
    tmp36 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp35, 0))
    tmp37 = tmp32 - tmp36
    tmp38 = tl.exp(tmp37)
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp41 = tl.where(rmask, tmp39, 0)
    tmp42 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tl.store(out_ptr0 + (r2 + (513*x3)), tmp31, rmask)
    tl.store(out_ptr1 + (x3), tmp36, None)
    tl.store(out_ptr2 + (x3), tmp42, None)
''')
