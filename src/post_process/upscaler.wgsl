@group(0) @binding(0)
var source_ao: texture_2d<f32>;
@group(0) @binding(1)
var depth: texture_2d<f32>;
@group(0) @binding(2)
var s: sampler;

const pos = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos[index], 0.0, 1.0);
}

fn gaussian(sigma: f32, x: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

const c_offset = array(
    vec2<f32>(0., 0.),
    vec2<f32>(0., 1.),
    vec2<f32>(1., 0.),
    vec2<f32>(1., 1.),
);

fn JoinedBilateralUpsample(p: vec2<f32>, buffer_size: vec2<f32>) -> f32 {
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

    let tex0 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[0], 0.0).r;
    let tex1 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[1], 0.0).r;
    let tex2 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[2], 0.0).r;
    let tex3 = textureSampleLevel(source_ao, s, pixel + c_texelSize * c_offset[3], 0.0).r;

    let sigmaV = 0.002;
    //    wXX = bilateral gaussian weight from depth * bilinear weight
    let w0 = gaussian(sigmaV, abs(I - Z0));
    let w1 = gaussian(sigmaV, abs(I - Z1));
    let w2 = gaussian(sigmaV, abs(I - Z2));
    let w3 = gaussian(sigmaV, abs(I - Z3));

    let tot_weight = (w0 + w1 + w2 + w3);
    let weighted_res = (w0 * tex0 + w1 * tex1 + w2 * tex2 + w3 * tex3) / tot_weight;
    return weighted_res;
}

@fragment
fn fs_main(@builtin(position) fcoords: vec4<f32>) -> @location(0) f32 {
    let buffer_size = vec2<f32>(textureDimensions(depth));
    return JoinedBilateralUpsample(fcoords.xy / buffer_size, buffer_size);
}
