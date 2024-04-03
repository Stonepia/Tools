

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/q3/cq3cxzz6wa7ezz2c2advdqdakxqfbosdrn7oyzjz3jxn4tuel42d.py
# Source Nodes: [iadd, softmax], Original ATen: [aten._softmax, aten._to_copy, aten.add]
# iadd => add_5
# softmax => amax, clone_5, convert_element_type_5, div_7, exp, sub_4, sum_1
triton_per_fused__softmax__to_copy_add_20 = async_compile.triton('triton_per_fused__softmax__to_copy_add_20', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel):
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
    x1 = (xindex // 12) % 1024
    r3 = rindex
    x0 = xindex % 12
    x2 = (xindex // 12288)
    x4 = xindex
    tmp16 = tl.load(in_ptr1 + (r3 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (525312*x0) + (6303744*x2)), rmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r3 + (513*(x1 % 256)) + (131328*((((256*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (x1 % 256)) // 256) % 4)) + (525312*x2)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 768, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to(r3, [RBLOCK])
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((-197632) + r3 + (257*x1)), rmask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = (tmp7 != 0)
    tmp9 = tl.load(in_ptr1 + (r3 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (525312*x0) + (6303744*x2)), rmask & tmp6, other=0.0).to(tl.float32)
    tmp10 = float("-inf")
    tmp11 = tl.where(tmp8, tmp10, tmp9)
    tmp12 = tl.where(tmp6, tmp11, 0.0)
    tmp13 = tl.load(in_ptr1 + (r3 + (513*(x1 % 256)) + (131328*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (525312*x0) + (6303744*x2)), rmask & tmp2, other=0.0).to(tl.float32)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = tl.where(tmp2, tmp14, 0.0)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = tl.load(in_ptr2 + ((-393984) + r3 + (513*x1) + (131328*x2)), rmask & tmp2, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp2, tmp18, 0.0)
    tmp20 = (256*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (x1 % 256)
    tmp21 = tmp20 < tmp4
    tmp22 = tl.full([1], 257, tl.int64)
    tmp23 = tmp3 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.load(in_ptr3 + (r3 + (257*(x1 % 256)) + (65792*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = (tmp25 != 0)
    tmp27 = tl.load(in_ptr4 + (r3 + (513*(x1 % 256)) + (131328*((((256*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (x1 % 256)) // 256) % 4)) + (525312*x2)), rmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tl.where(tmp26, tmp10, tmp27)
    tmp29 = tl.where(tmp24, tmp28, 0.0)
    tmp30 = tl.load(in_ptr4 + (r3 + (513*(x1 % 256)) + (131328*((((256*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4)) + (x1 % 256)) // 256) % 4)) + (525312*x2)), rmask & tmp21, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp23, tmp29, tmp30)
    tmp32 = tl.where(tmp21, tmp31, 0.0)
    tmp34 = tl.where(tmp21, tmp32, tmp33)
    tmp35 = tl.where(tmp2, tmp19, tmp34)
    tmp36 = tmp17 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp40 = tl.where(rmask, tmp38, float("-inf"))
    tmp41 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp40, 0))
    tmp42 = tmp37 - tmp41
    tmp43 = tl.exp(tmp42)
    tmp44 = tl.broadcast_to(tmp43, [RBLOCK])
    tmp46 = tl.where(rmask, tmp44, 0)
    tmp47 = triton_helpers.promote_to_tensor(tl.sum(tmp46, 0))
    tmp48 = tmp43 / tmp47
    tl.store(out_ptr3 + (r3 + (513*x4)), tmp48, rmask)
''')
