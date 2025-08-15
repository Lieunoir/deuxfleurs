struct Params {
    x_mul: f32,
    x_add: f32,
    y_mul: f32,
    y_add: f32,
}

@group(0) @binding(0)
var t_a: texture_2d<f32>;
@group(0) @binding(1)
var t_n: texture_2d<f32>;
@group(0) @binding(2)
var t_v: texture_2d<f32>;
@group(0) @binding(3)
var s: sampler;
@group(0) @binding(4)
var<uniform> param: Params;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

const pos = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos[index], 0.0, 1.0);
    return out;
}

const PI: f32 = 3.14159265359;
const F0 = vec3<f32>(0.04, 0.04, 0.04);
const UP = vec3<f32>(0., 1., 0.);
const RIGHT = vec3<f32>(-1., 0., 0.);
const FORWARD = vec3<f32>(0., 0., -1.);
const LIGHT_DIR = (RIGHT - UP - FORWARD) / sqrt(3.);
const LIGHT_DIR_2 = (-RIGHT + UP - FORWARD) / sqrt(3.);
const LIGHT_DIR_3 = (RIGHT + UP + FORWARD) / sqrt(3.);

// PBR functions taken from https://learnopengl.com/PBR/Theory
fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, a: f32) -> f32 {
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;

    let nom = a2;
    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = denom * denom;

    return nom / denom;
}

fn GeometrySchlickGGX(NdotV: f32, k: f32) -> f32 {
    let nom = NdotV;
    let denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

fn fresnelSchlick(cosTheta: f32) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

fn normalized_view_from_screen_coord(coord: vec2<f32>) -> vec3<f32> {
    // reconstruct view-space position from the screen coordinate and view space depth.
    return normalize(vec3<f32>(
        vec2<f32>(param.x_mul, param.y_mul) * coord + vec2<f32>(param.x_add, param.y_add),
        1.
    ));
}

@fragment
fn fs_main(@builtin(position) fcoords: vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(fcoords.xy);
    let albedo = textureLoad(t_a, coords, 0);
    if albedo.w < 0.01 {
        discard;
    }
    let buffer_size = textureDimensions(t_a);
    let view_dir = normalized_view_from_screen_coord(fcoords.xy / vec2<f32>(buffer_size));
    let normal = normalize(textureLoad(t_n, coords, 0).xyz * 2. - vec3<f32>(1.));
    let visibility = textureLoad(t_v, coords, 0).x;

    let kd = 1. * albedo.xyz;
    let k = albedo.w;

    let n_dot_v = max(dot(normal, view_dir), 0.);
    let f = fresnelSchlick(dot(view_dir, normal));
    let ggx1 = GeometrySchlickGGX(n_dot_v, k);
    let f_ct_fact = select(vec3<f32>(0.), ggx1 * f / (4. * n_dot_v), n_dot_v > 0.);

    let half_dir = normalize(view_dir + LIGHT_DIR);
    let d = DistributionGGX(normal, half_dir, k);
    let ggx2 = GeometrySchlickGGX(max(dot(view_dir, LIGHT_DIR), 0.), k);
    let f_ct = d * f_ct_fact * ggx2;
    var result = 0.55 * (kd * max(dot(normal, LIGHT_DIR), 0.0) + f_ct);

    let half_dir_2 = normalize(view_dir + LIGHT_DIR_2);
    let d2 = DistributionGGX(normal, half_dir_2, albedo.w);
    let ggx2_2 = GeometrySchlickGGX(max(dot(view_dir, LIGHT_DIR_2), 0.), k);
    let f_ct_2 = d2 * f_ct_fact * ggx2_2;
    result += 1.6 * (kd * max(dot(normal, LIGHT_DIR_2), 0.0) + f_ct_2);

    let half_dir_3 = normalize(view_dir + LIGHT_DIR_3);
    let d3 = DistributionGGX(normal, half_dir_3, albedo.w);
    let ggx2_3 = GeometrySchlickGGX(max(dot(view_dir, LIGHT_DIR_3), 0.), k);
    let f_ct_3 = d3 * f_ct_fact * ggx2_3;
    result += 1.4 * (kd * max(dot(normal, LIGHT_DIR_3), 0.0) + f_ct_3);

    result *= 1.2 * visibility;

	//Tone mapping
    let m1 = mat3x3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777,
    );
    let m2 = mat3x3(
        1.60475, -0.10208, -0.00327,
        -0.53108, 1.10813, -0.07276,
        -0.07367, -0.00605, 1.07602,
    );
    let v = m1 * result;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return vec4<f32>(clamp(m2 * (a / b), vec3(0.0), vec3(1.0)), 1.0);
}
