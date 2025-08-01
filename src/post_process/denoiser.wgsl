struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct OldCamera {
    view_proj: mat4x4<f32>,
}

@group(1) @binding(0)
var source_ao: texture_2d<f32>;
@group(1) @binding(1)
var source_edges: texture_2d<f32>;
@group(1) @binding(2)
var history: texture_2d<f32>;
@group(1) @binding(3)
var depth: texture_2d<f32>;
@group(1) @binding(4)
var old_depth: texture_2d<f32>;
@group(1) @binding(5)
var s: sampler;
@group(1) @binding(6)
var<uniform> old_camera: OldCamera;

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

fn add_sample(ssaoValue: f32, edgeValue: f32, sum: ptr<function, f32>, sumWeight: ptr<function, f32>) {
    let weight = edgeValue;

    *sum += (weight * ssaoValue);
    *sumWeight += weight;
}

fn delinearize_depth(depth: f32) -> f32 {
    return 0.0008740438 / depth + 1.0001;
}

fn get_previous_value(uv: vec2<f32>) -> vec2<f32> {
    let depth_value = delinearize_depth(textureSampleLevel(depth, s, uv, 0.).x);
    let pos_clip = vec4(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, depth_value, 1.0);
    let pos_world_w = camera.view_inv * pos_clip;
    let pos_world = pos_world_w / pos_world_w.wwww;
    let recovered_clip_w = old_camera.view_proj * pos_world;
    let recovered_clip_z = - recovered_clip_w.w;
    let recovered_clip = recovered_clip_w.xyz / recovered_clip_w.www;
    let recovered_uv = vec2<f32>(0.5 * recovered_clip.x + 0.5, - 0.5 * recovered_clip.y + 0.5);
    let old_ao = textureSample(history, s, recovered_uv).x;
    let linear_old_depth_sample = textureSampleLevel(old_depth, s, recovered_uv, 0.).x;
    let old_depth_sample = delinearize_depth(linear_old_depth_sample);
    let weight = select(0., 1., abs(linear_old_depth_sample - recovered_clip_z) < 0.005);
    return vec2<f32>(old_ao, weight);
}

fn gaussian(sigma: f32, x: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

const c_offset = array(
    vec2<f32>(0., 0.),
    vec2<f32>(0., 1.),
    vec2<f32>(1., 0.),
    vec2<f32>(0., -1.),
    vec2<f32>(-1., 0.),
    vec2<f32>(1., 1.),
    vec2<f32>(1., -1.),
    vec2<f32>(-1., 1.),
    vec2<f32>(-1., -1.),
);

fn JoinedBilateralUpsample(p: vec2<f32>, buffer_size: vec2<f32>) -> f32 { // based on: https://johanneskopf.de/publications/jbu/paper/FinalPaper_0185.pdf
  //           https://bartwronski.com/2019/09/22/local-linear-models-guided-filter/
  //		   https://www.shadertoy.com/view/MllSzX

    let half_p = 0.5 * p;
    let c_textureSize = buffer_size;
    let c_texelSize = 2.0 / c_textureSize;
    var pixel = half_p * c_textureSize;// + 0.5;
    //var pixel = p * buffer_size;
    let f = fract(pixel);
    //pixel = (floor(pixel) / c_textureSize) - vec2<f32>(c_texelSize / 2.0);
    pixel = p - vec2<f32>(c_texelSize / 2.0);

    let I = textureSampleLevel(depth, s, p, 0.0).r;

    let Z0 = textureSampleLevel(depth, s, pixel + c_texelSize * c_offset[0], 1.0).r;
    let Z1 = textureSampleLevel(depth, s, pixel + c_texelSize * c_offset[1], 1.0).r;
    let Z2 = textureSampleLevel(depth, s, pixel + c_texelSize * c_offset[2], 1.0).r;
    let Z3 = textureSampleLevel(depth, s, pixel + c_texelSize * c_offset[3], 1.0).r;
    let Z4 = textureSampleLevel(depth, s, pixel + c_texelSize * c_offset[4], 1.0).r;
    let Z5 = textureSampleLevel(depth, s, pixel + c_texelSize * c_offset[5], 1.0).r;
    let Z6 = textureSampleLevel(depth, s, pixel + c_texelSize * c_offset[6], 1.0).r;
    let Z7 = textureSampleLevel(depth, s, pixel + c_texelSize * c_offset[7], 1.0).r;
    let Z8 = textureSampleLevel(depth, s, pixel + c_texelSize * c_offset[8], 1.0).r;

    let tex0 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[0], 0.0).r;
    let tex1 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[1], 0.0).r;
    let tex2 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[2], 0.0).r;
    let tex3 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[3], 0.0).r;
    let tex4 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[4], 0.0).r;
    let tex5 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[5], 0.0).r;
    let tex6 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[6], 0.0).r;
    let tex7 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[7], 0.0).r;
    let tex8 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[8], 0.0).r;

    let sigmaV = 0.002;
    //    wXX = bilateral gaussian weight from depth * bilinear weight
    let w0 = gaussian(sigmaV, abs(I - Z0));
    let w1 = gaussian(sigmaV, abs(I - Z1));
    let w2 = gaussian(sigmaV, abs(I - Z2));
    let w3 = gaussian(sigmaV, abs(I - Z3));
    let w4 = gaussian(sigmaV, abs(I - Z4));
    let w5 = gaussian(sigmaV, abs(I - Z5));
    let w6 = gaussian(sigmaV, abs(I - Z6));
    let w7 = gaussian(sigmaV, abs(I - Z7));
    let w8 = gaussian(sigmaV, abs(I - Z8));

    let tot_weight = (w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8);
    let weighted_res = (w0 * tex0 + w1 * tex1 + w2 * tex2 + w3 * tex3 + w4 * tex4 + w5 * tex5 + w6 * tex6 + w7 * tex7 + w8 * tex8) / tot_weight;
    //return select(tex0, weighted_res, tot_weight > 0.01);
    return weighted_res;
    //return tex00;
}

@fragment
fn fs_main(@builtin(position) fcoords: vec4<f32>) -> @location(0) f32 {
    let buffer_size = vec2<f32>(textureDimensions(depth));
    return JoinedBilateralUpsample(fcoords.xy / buffer_size, buffer_size);
    /*
    let buffer_size_inv = 1. / vec2<f32>(textureDimensions(depth));
    let gather_offset = - vec2<f32>(0.25) * buffer_size_inv;
    let gatherCenter = fcoords.xy * buffer_size_inv + gather_offset;

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

    add_sample(ssaoValueL, edgesC_LRTB.x, &sum, &sumWeight);
    add_sample(ssaoValueR, edgesC_LRTB.y, &sum, &sumWeight);
    add_sample(ssaoValueT, edgesC_LRTB.z, &sum, &sumWeight);
    add_sample(ssaoValueB, edgesC_LRTB.w, &sum, &sumWeight);

    add_sample(ssaoValueTL, weightTL, &sum, &sumWeight);
    add_sample(ssaoValueTR, weightTR, &sum, &sumWeight);
    add_sample(ssaoValueBL, weightBL, &sum, &sumWeight);
    add_sample(ssaoValueBR, weightBR, &sum, &sumWeight);

    let blurred_ao = sum / sumWeight;

    var min_ssao = ssaoValue;
    min_ssao = select(min_ssao, min(min_ssao, ssaoValueL), edgesC_LRTB.x > 0.8);
    min_ssao = select(min_ssao, min(min_ssao, ssaoValueR), edgesC_LRTB.y > 0.8);
    min_ssao = select(min_ssao, min(min_ssao, ssaoValueT), edgesC_LRTB.z > 0.8);
    min_ssao = select(min_ssao, min(min_ssao, ssaoValueB), edgesC_LRTB.w > 0.8);
    var max_ssao = ssaoValue;
    max_ssao = select(max_ssao, max(max_ssao, ssaoValueL), edgesC_LRTB.x > 0.8);
    max_ssao = select(max_ssao, max(max_ssao, ssaoValueR), edgesC_LRTB.y > 0.8);
    max_ssao = select(max_ssao, max(max_ssao, ssaoValueT), edgesC_LRTB.z > 0.8);
    max_ssao = select(max_ssao, max(max_ssao, ssaoValueB), edgesC_LRTB.w > 0.8);
    var weight = 0.9;
    weight = select(weight, 0.5 * (previous_ao.x - blurred_ao), max_ssao < previous_ao.x);
    weight = select(weight, 0.5 * (blurred_ao - previous_ao.x), min_ssao > previous_ao.x);
    previous_ao.x = min(max_ssao, previous_ao.x);
    previous_ao.x = max(min_ssao, previous_ao.x);
    weight *= previous_ao.y;


    /*
    let depth_value = exp(textureSampleLevel(depth, s, fcoords.xy / vec2<f32>(buffer_size), 0.).x);
    let depth_c = exp(textureSampleLevel(depth, s, fcoords.xy / vec2<f32>(buffer_size), 1.).x);
    let depth_l = exp(textureSampleLevel(depth, s, fcoords.xy / vec2<f32>(buffer_size), 1., vec2<i32>(-1, 0)).x);
    let depth_t = exp(textureSampleLevel(depth, s, fcoords.xy / vec2<f32>(buffer_size), 1., vec2<i32>(0, 1)).x);
    let depth_r = exp(textureSampleLevel(depth, s, fcoords.xy / vec2<f32>(buffer_size), 1., vec2<i32>(1, 0)).x);
    let depth_b = exp(textureSampleLevel(depth, s, fcoords.xy / vec2<f32>(buffer_size), 1., vec2<i32>(0, -1)).x);

    let ao_c = exp(textureSample(source_ao, s, fcoords.xy / vec2<f32>(buffer_size)).x);
    let ao_l = exp(textureSample(source_ao, s, fcoords.xy / vec2<f32>(buffer_size), vec2<i32>(-1, 0)).x);
    let ao_t = exp(textureSample(source_ao, s, fcoords.xy / vec2<f32>(buffer_size), vec2<i32>(0, 1)).x);
    let ao_r = exp(textureSample(source_ao, s, fcoords.xy / vec2<f32>(buffer_size), vec2<i32>(1, 0)).x);
    let ao_b = exp(textureSample(source_ao, s, fcoords.xy / vec2<f32>(buffer_size), vec2<i32>(0, -1)).x);
    let depth_d_c = abs(depth_c - depth_value);
    let depth_d_l = abs(depth_l - depth_value);
    let depth_d_t = abs(depth_t - depth_value);
    let depth_d_r = abs(depth_r - depth_value);
    let depth_d_b = abs(depth_b - depth_value);
    var closest_depth_d = depth_d_c;
    var closest_ao = ao_c;
    closest_ao = select(closest_ao, ao_l, depth_d_l < closest_depth_d);
    closest_depth_d = select(closest_depth_d, depth_d_l, depth_d_l < closest_depth_d);
    closest_ao = select(closest_ao, ao_t, depth_d_t < closest_depth_d);
    closest_depth_d = select(closest_depth_d, depth_d_t, depth_d_t < closest_depth_d);
    closest_ao = select(closest_ao, ao_r, depth_d_r < closest_depth_d);
    closest_depth_d = select(closest_depth_d, depth_d_r, depth_d_r < closest_depth_d);
    closest_ao = select(closest_ao, ao_b, depth_d_b < closest_depth_d);
    closest_depth_d = select(closest_depth_d, depth_d_b, depth_d_b < closest_depth_d);
    return previous_ao.y * previous_ao.x + (1. - previous_ao.y) * closest_ao;
    */

    return weight * previous_ao.x + (1. - weight) * blurred_ao;
    */
}
