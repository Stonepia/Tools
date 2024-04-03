

# Original file: ./XLNetLMHeadModel__0_forward_565.0/XLNetLMHeadModel__0_forward_565.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/vs/cvsse72cfxx7bdg3zywomvuuvdkfzii4ivhj7z56lpiau7ix3wtm.py
# Source Nodes: [l__mod___transformer_dropout, l__mod___transformer_word_embedding], Original ATen: [aten.embedding, aten.native_dropout, aten.transpose]
# l__mod___transformer_dropout => gt, mul, mul_1
# l__mod___transformer_word_embedding => embedding
triton_poi_fused_embedding_native_dropout_transpose_1 = async_compile.triton('triton_poi_fused_embedding_native_dropout_transpose_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*bf16', 3: '*i1', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_native_dropout_transpose_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_embedding_native_dropout_transpose_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 1024)
    x1 = xindex % 1024
    tmp7 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tl.where(tmp7 < 0, tmp7 + 32000, tmp7)
    # tl.device_assert((0 <= tmp8) & (tmp8 < 32000), "index out of bounds: 0 <= tmp8 < 32000")
    tmp9 = tl.load(in_ptr2 + (x1 + (1024*tmp8)), None).to(tl.float32)
    tmp10 = tmp6 * tmp9
    tmp11 = 1.1111111111111112
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr1 + (x0), tmp5, None)
    tl.store(out_ptr2 + (x0), tmp12, None)
    tl.store(out_ptr3 + (x0), tmp12, None)
''')
