struct Parameters {
    reprojection: mat4x4<f32>,
    depth_mul: f32,
    depth_add: f32,
    x_mul: f32,
    x_add: f32,
    y_mul: f32,
    y_add: f32,
    frame_index: u32,
    world_distance: f32,
    pix_dif: vec2<f32>,
}

@group(0) @binding(0)
var source_ao: texture_2d<f32>;
@group(0) @binding(1)
var source_edges: texture_2d<f32>;
@group(0) @binding(2)
var history: texture_2d<f32>;
@group(0) @binding(3)
var depth: texture_2d<f32>;
@group(0) @binding(4)
var old_depth: texture_2d<f32>;
@group(0) @binding(5)
var s: sampler;
@group(0) @binding(6)
var<uniform> param: Parameters;


const pos = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos[index], 0.0, 1.0);
}

fn unpack_edges(packedValIn: f32) -> vec4<f32> {
    let packedVal = u32(packedValIn * 255.5);
    var edgesLRTB: vec4<f32>;
    edgesLRTB.x = f32((packedVal >> 6) & 0x03) / 3.0;
    edgesLRTB.y = f32((packedVal >> 4) & 0x03) / 3.0;
    edgesLRTB.z = f32((packedVal >> 2) & 0x03) / 3.0;
    edgesLRTB.w = f32((packedVal >> 0) & 0x03) / 3.0;

    return saturate(edgesLRTB);
}

fn add_sample(ssaoValue: f32, edgeValue: f32, sum: ptr<function, f32>, sumWeight: ptr<function, f32>, sum_squared: ptr<function, f32>) {
    let weight = edgeValue;

    *sum += weight * ssaoValue;
    *sumWeight += weight;
    *sum_squared += weight * ssaoValue * ssaoValue;
}

fn delinearize_depth(depth: f32) -> f32 {
    return param.depth_mul / depth - param.depth_add;
}

fn linearize_depth(depth: f32) -> f32 {
    return param.depth_mul / (param.depth_add + depth);
}

fn get_previous_value(uv: vec2<f32>) -> vec2<f32> {
    let depth_value = delinearize_depth(textureSampleLevel(depth, s, uv, 0.).x);
    let pos_clip = vec4(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, depth_value, 1.0);
    let recovered_clip_w = param.reprojection * pos_clip;
    let recovered_clip = recovered_clip_w.xyz / recovered_clip_w.www;
    let recovered_clip_z = linearize_depth(recovered_clip.z);
    let recovered_uv = vec2<f32>(0.5 * recovered_clip.x + 0.5, - 0.5 * recovered_clip.y + 0.5);
    let old_ao = textureSample(history, s, recovered_uv).x;
    let linear_old_depth_sample = textureSampleLevel(old_depth, s, recovered_uv, 0.).x;
    let weight = select(0., 1., abs(1. - linear_old_depth_sample / recovered_clip_z) < 0.05);
    //let weight = 1.;
    return vec2<f32>(old_ao, weight);
}

@fragment
fn fs_main(@builtin(position) fcoords: vec4<f32>) -> @location(0) f32 {
    let buffer_size_inv = param.pix_dif;
    let gatherCenter = (fcoords.xy - vec2<f32>(0.25)) * buffer_size_inv;

    var previous_ao = get_previous_value(fcoords.xy * buffer_size_inv);

    let blurAmount = 1.2f;
    let diagWeight = 0.65 * 0.5;

    // gather edge and visibility quads, used later
    let edgesQ0 = textureGather(0, source_edges, s, gatherCenter, vec2<i32>(0, 0));
    let edgesQ1 = textureGather(0, source_edges, s, gatherCenter, vec2<i32>(2, 0));
    let edgesQ2 = textureGather(0, source_edges, s, gatherCenter, vec2<i32>(1, 2));

    let visQ0 = textureGather(0, source_ao, s, gatherCenter, vec2<i32>(0, 0));
    let visQ1 = textureGather(0, source_ao, s, gatherCenter, vec2<i32>(2, 0));
    let visQ2 = textureGather(0, source_ao, s, gatherCenter, vec2<i32>(0, 2));
    let visQ3 = textureGather(0, source_ao, s, gatherCenter, vec2<i32>(2, 2));

    let edgesL_LRTB = unpack_edges(edgesQ0.x);
    let edgesT_LRTB = unpack_edges(edgesQ0.z);
    let edgesR_LRTB = unpack_edges(edgesQ1.x);
    let edgesB_LRTB = unpack_edges(edgesQ2.w);
    var edgesC_LRTB = unpack_edges(edgesQ0.y);

    // Edges aren't perfectly symmetrical: edge detection algorithm does not guarantee that a left edge on the right pixel will match the right edge on the left pixel (although
    // they will match in majority of cases). This line further enforces the symmetricity, creating a slightly sharper blur. Works real nice with TAA.
    edgesC_LRTB *= vec4<f32>(edgesL_LRTB.y, edgesR_LRTB.x, edgesT_LRTB.w, edgesB_LRTB.z);

    let leak_threshold = 2.5;
    let leak_strength = 0.5;
    let edginess = (saturate(4.0 - leak_threshold - dot(edgesC_LRTB, vec4<f32>(1.))) / (4 - leak_threshold)) * leak_strength;
    edgesC_LRTB = saturate(edgesC_LRTB + edginess);

    // for diagonals; used by first and second pass
    let weightTL = diagWeight * (edgesC_LRTB.x * edgesL_LRTB.z + edgesC_LRTB.z * edgesT_LRTB.x);
    let weightTR = diagWeight * (edgesC_LRTB.z * edgesT_LRTB.y + edgesC_LRTB.y * edgesR_LRTB.z);
    let weightBL = diagWeight * (edgesC_LRTB.w * edgesB_LRTB.x + edgesC_LRTB.x * edgesL_LRTB.w);
    let weightBR = diagWeight * (edgesC_LRTB.y * edgesR_LRTB.w + edgesC_LRTB.w * edgesB_LRTB.y);

    // first pass
    let ssaoValue = visQ0.y;
    let ssaoValueL = visQ0.x;
    let ssaoValueT = visQ0.z;
    let ssaoValueR = visQ1.x;
    let ssaoValueB = visQ2.z;
    let ssaoValueTL = visQ0.w;
    let ssaoValueBR = visQ3.w;
    let ssaoValueTR = visQ1.w;
    let ssaoValueBL = visQ2.w;

    var sumWeight = blurAmount;
    var sum = ssaoValue * sumWeight;
    var sum_squared = ssaoValue * ssaoValue * sumWeight;

    add_sample(ssaoValueL, edgesC_LRTB.x, &sum, &sumWeight, &sum_squared);
    add_sample(ssaoValueR, edgesC_LRTB.y, &sum, &sumWeight, &sum_squared);
    add_sample(ssaoValueT, edgesC_LRTB.z, &sum, &sumWeight, &sum_squared);
    add_sample(ssaoValueB, edgesC_LRTB.w, &sum, &sumWeight, &sum_squared);

    add_sample(ssaoValueTL, weightTL, &sum, &sumWeight, &sum_squared);
    add_sample(ssaoValueTR, weightTR, &sum, &sumWeight, &sum_squared);
    add_sample(ssaoValueBL, weightBL, &sum, &sumWeight, &sum_squared);
    add_sample(ssaoValueBR, weightBR, &sum, &sumWeight, &sum_squared);

    let blurred_ao = sum / sumWeight;

    //let avg_ssao = (ssaoValue + ssaoValueL + ssaoValueB + ssaoValueR + ssaoValueT) * 0.2;
    //let avg_squared_ssao = (ssaoValue * ssaoValue + ssaoValueL * ssaoValueL + ssaoValueB * ssaoValueB + ssaoValueR * ssaoValueR + ssaoValueT * ssaoValueT) * 0.2;
    let avg_ssao = blurred_ao;
    let avg_squared_ssao = sum_squared / sumWeight;
    let std_diff = sqrt(avg_squared_ssao - avg_ssao * avg_ssao);
    let min_ssao = avg_ssao - std_diff;
    let max_ssao = avg_ssao + std_diff;

    //var min_ssao = ssaoValue;
    //min_ssao = select(min_ssao, min(min_ssao, ssaoValueL), edgesC_LRTB.x > 0.8);
    //min_ssao = select(min_ssao, min(min_ssao, ssaoValueR), edgesC_LRTB.y > 0.8);
    //min_ssao = select(min_ssao, min(min_ssao, ssaoValueT), edgesC_LRTB.z > 0.8);
    //min_ssao = select(min_ssao, min(min_ssao, ssaoValueB), edgesC_LRTB.w > 0.8);
    //var max_ssao = ssaoValue;
    //max_ssao = select(max_ssao, max(max_ssao, ssaoValueL), edgesC_LRTB.x > 0.8);
    //max_ssao = select(max_ssao, max(max_ssao, ssaoValueR), edgesC_LRTB.y > 0.8);
    //max_ssao = select(max_ssao, max(max_ssao, ssaoValueT), edgesC_LRTB.z > 0.8);
    //max_ssao = select(max_ssao, max(max_ssao, ssaoValueB), edgesC_LRTB.w > 0.8);
    var weight = 0.9;
    //weight = select(weight, 0.5 * (previous_ao.x - blurred_ao), max_ssao < previous_ao.x);
    //weight = select(weight, 0.5 * (blurred_ao - previous_ao.x), min_ssao > previous_ao.x);
    previous_ao.x = min(max_ssao, previous_ao.x);
    previous_ao.x = max(min_ssao, previous_ao.x);
    weight *= previous_ao.y;
    return (weight * previous_ao.x + (1. - weight) * blurred_ao);
}
