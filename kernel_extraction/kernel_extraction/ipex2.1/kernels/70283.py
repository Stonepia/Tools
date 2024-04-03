

# Original file: ./MobileBertForMaskedLM__0_backward_210.1/MobileBertForMaskedLM__0_backward_210.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/aa/caaq7cdj7lr4st6ftlkz5dstxcgbxgxheccx2wczhjtmunqbczfq.py
# Source Nodes: [add_348, add_351, add_352, add_353, add_354, add_355, add_356, add_357, mul_186, mul_188, mul_189, mul_190], Original ATen: [aten.add, aten.mul]
# add_348 => add_348
# add_351 => add_351
# add_352 => add_352
# add_353 => add_353
# add_354 => add_354
# add_355 => add_355
# add_356 => add_356
# add_357 => add_357
# mul_186 => mul_232
# mul_188 => mul_236
# mul_189 => mul_237
# mul_190 => mul_238
triton_poi_fused_add_mul_11 = async_compile.triton('triton_poi_fused_add_mul_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr7 + (x2), None).to(tl.float32)
    tmp14 = tl.load(in_ptr8 + (x2), None).to(tl.float32)
    tmp15 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tl.load(in_ptr13 + (x2), None).to(tl.float32)
    tmp26 = tl.load(in_ptr14 + (x2), None).to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp1 + tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp0 + tmp11
    tmp16 = tmp12 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp14 + tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tmp13 + tmp23
    tmp27 = tmp25 + tmp26
    tmp28 = tmp27 * tmp20
    tl.store(out_ptr0 + (x2), tmp12, None)
    tl.store(out_ptr1 + (x2), tmp24, None)
    tl.store(out_ptr2 + (x2), tmp28, None)
''')
