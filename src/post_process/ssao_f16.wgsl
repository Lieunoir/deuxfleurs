enable f16;

struct Parameters {
    reprojection: mat4x4<f32>,
    depth_mul: f32,
    depth_add: f32,
    x_mul: f32,
    x_add: f32,
    y_mul: f32,
    y_add: f32,
    frame_index: u32,
    world_distance_inv: f32,
    pix_dif: vec2<f32>,
}

@group(0) @binding(0)
var t_n: texture_2d<f32>;
@group(0) @binding(1)
var t_d: texture_2d<f32>;
@group(0) @binding(2)
var s: sampler;
@group(0) @binding(3)
var<uniform> param: Parameters;
@group(0) @binding(4)
var hilbert: texture_2d<u32>;
@group(0) @binding(5)
var noise_s: sampler;

const pos = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos[index], 0.0, 1.0);
}

const PI: f16 = 3.14159265359;
fn view_from_screen_coord(local_param: Parameters, coord: vec2<f32>, linear_depth_sample: f16) -> vec3<f32> {
    // reconstruct view-space position from the screen coordinate and view space depth.
    return vec3<f32>(
        (vec2<f32>(local_param.x_mul, local_param.y_mul) * coord
          + vec2<f32>(local_param.x_add, local_param.y_add)) * f32(linear_depth_sample),
        f32(linear_depth_sample)
    );
}

fn fast_sqrt(x: f32) -> f32 {
    return bitcast<f32>(0x1FBD1DF5 + (bitcast<i32>(x) >> 1));
}

fn fast_acos(x: f16) -> f16 {
    let abs_x = abs(x);
    var res = 1.5707963267948966 - 0.1565827644218014 * abs_x;
    res *= f16(fast_sqrt(f32(1.0 - abs_x)));
    return select(PI - res, res, x >= 0);
}

const g: f32 = 1.32471795724474602596;
const a1: f32 = 1. - 1.0 / g;
const a2: f32 = 1. - 1.0 / (g * g);

fn calculate_edges(centerZ: f16, leftZ: f16, rightZ: f16, topZ: f16, bottomZ: f16) -> vec4<f16> {
    var edgesLRTB = vec4<f16>(leftZ, rightZ, topZ, bottomZ) - centerZ;

    let slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
    let slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
    let edgesLRTBSlopeAdjusted = edgesLRTB + vec4<f16>(slopeLR, -slopeLR, slopeTB, -slopeTB);
    edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
    //remember: centerZ < 0.
    return vec4<f16>(saturate((1.25 + edgesLRTB / (centerZ * 0.011))));
}

// packing/unpacking for edges; 2 bits per edge mean 4 gradient values (0, 0.33, 0.66, 1) for smoother transitions!
fn pack_edges(in: vec4<f16>) -> f16 {
    var edgesLRTB = round(in * 2.9);
    return dot(edgesLRTB, vec4<f16>(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

const sample_factor: f16 = 1.;

struct FragmentOutput {
    @location(0) ssao: f32,
    @location(1) edges: f32,
}

const kernel_size = 2.;
const samples_per_slice: f16 = 3.;
const phi_mul = PI / kernel_size;
const cos_phi_mul = cos(phi_mul);
const sin_phi_mul = sin(phi_mul);
const cos_2_phi_mul = cos(2. * phi_mul);
const sin_2_phi_mul = sin(2. * phi_mul);
const vis_scale = 0.25 / kernel_size;
const max_radius_pix = 128. / sample_factor;
override max_mip_level: f16 = 0.;

const phi1 = 2654435769u;
const phi2 = vec2<u32>(3242174889u, 2447445413u);
const u_to_f_c = 1.0 / 4294967296.0;

fn float01(x: u32) -> f32 {
    return f32(x) * u_to_f_c;
}

fn v2_float01(x: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(x) * u_to_f_c;
}

fn get_sample(local_param: Parameters, origin: vec2<f16>, position: vec3<f32>, view_dir: vec3<f16>, sin_n: f16, noise: u32, iter_i: f16, iter_j: f16, dir: vec2<f16>, radius: f16, pix_dif: vec2<f16>, world_radius_mul: f16) -> vec2<f16> {
    let step_noise = f16(float01(noise + phi1 * u32(iter_i * samples_per_slice + iter_j)));
    //let step_noise = fract(noise + (iter_i * samples_per_slice + iter_j) * 0.6180339887498948482f);
    let sample_offset_length = (step_noise + iter_j) / samples_per_slice * radius;
    // compile time version, so mip_level is known at compile time too
    let sample_offset_length_approx = (0.5 + iter_j) / samples_per_slice * max_radius_pix;
    let sample_offset = sample_offset_length * pix_dif * dir;
    let mip_level = clamp(log2(sample_offset_length_approx) - 3.3, 0., max_mip_level);
    //let mip_level = max_mip_level;
    //let mip_level = 0.;

    let sample_coords_1 = vec2<f32>(origin + sample_offset);
    let sample_coords_2 = vec2<f32>(origin - sample_offset);
    let sample_depth_1 = f16(textureSampleLevel(t_d, s, sample_coords_1, f32(mip_level)).x);
    let sample_depth_2 = f16(textureSampleLevel(t_d, s, sample_coords_2, f32(mip_level)).x);

    let sample_1 = view_from_screen_coord(local_param, sample_coords_1, sample_depth_1);
    let sample_2 = view_from_screen_coord(local_param, sample_coords_2, sample_depth_2);
    let dir_1 = vec3<f16>(sample_1 - position);
    let dir_2 = vec3<f16>(sample_2 - position);

    //Maybe too much precision loss here
    let d_s_1_norm_inv = inverseSqrt(dot(dir_1, dir_1));
    let d_s_2_norm_inv = inverseSqrt(dot(dir_2, dir_2));

    let d_s_1 = dir_1 * d_s_1_norm_inv;
    let d_s_2 = dir_2 * d_s_2_norm_inv;

    let l_s_1 = saturate(world_radius_mul / d_s_1_norm_inv - 5.);
    let l_s_2 = saturate(world_radius_mul / d_s_2_norm_inv - 5.);
    let dot_s_1 = mix(dot(d_s_1, view_dir), -sin_n, l_s_1);
    let dot_s_2 = mix(dot(d_s_2, view_dir), sin_n, l_s_2);
    //let dot_s_1 = dot(d_s_1, view_dir);
    //let dot_s_2 = dot(d_s_2, view_dir);
    return vec2<f16>(dot_s_1, dot_s_2);
}

@fragment
fn fs_main(@builtin(position) fcoords: vec4<f32>) -> FragmentOutput {
    let local_param = param;
    var out: FragmentOutput;
    let pix_dif = vec2<f16>(local_param.pix_dif);
    let frame_index = local_param.frame_index;
    let origin = sample_factor * vec2<f16>(fcoords.xy) * pix_dif;
    let coords = vec2<u32>(fcoords.xy);
    let normal_sample = textureLoad(t_n, coords, 0).xyz;
    //if ((coords.x + coords.y) % 2 == frame_index % 2) {
    //    discard;
    //}
    var noise_x = textureLoad(hilbert, coords % 16, 0).x;
    let gather_offset = vec2<f32>(origin - 0.25 * pix_dif);
    let values_ul = vec4<f16>(textureGather(0, t_d, s, gather_offset));
    let values_br = vec4<f16>(textureGather(0, t_d, s, gather_offset, vec2<i32>(1, 1)));
    let depth = values_ul.y;

    // viewspace Zs left top right bottom
    let pix_lz = values_ul.x;
    let pix_tz = values_ul.z;
    let pix_rz = values_br.z;
    let pix_bz = values_br.x;

    let edgesLRTB = calculate_edges(
        depth,
        pix_lz,
        pix_rz,
        pix_tz,
        pix_bz
    );
    let packed_edges = pack_edges(edgesLRTB);
    out.edges = f32(packed_edges);

    let position = view_from_screen_coord(local_param, vec2<f32>(origin), depth);
    let normal = normalize(vec3<f16>(normal_sample) * 2. - vec3<f16>(1.));
    let view_dir = vec3<f16>(normalize(-position));
    let n_cross_v = cross(normal, view_dir).xy;
    let cos_n_scaled = dot(view_dir, normal);

    //if abs(normal_sample[0]) + abs(normal_sample[1]) + abs(normal_sample[2]) < 0.01 {
    //    discard;
    //}

    // GTAO
    var visibility: f16 = 0.0;
    let world_radius_mul = -100. * f16(local_param.world_distance_inv);
    let wanted_screen_radius = 2. / (depth * f16(local_param.world_distance_inv));
    let radius = max_radius_pix * min(wanted_screen_radius, 1.);

    noise_x += local_param.frame_index;
    let noise = phi2 * noise_x;
    //let noise = vec2<f32>(fract(0.5 + vec2<f32>(a1, a2) * noise_x));
    let phi_init = f16(float01(noise.x)) * phi_mul;
    let cos_phi_init = cos(phi_init);
    let sin_phi_init = sin(phi_init);
    //for (var i = 0.; i < kernel_size; i += 1.)
    {
        let cos_phi = cos_phi_init;
        let sin_phi = sin_phi_init;
        let dir = vec2<f16>(cos_phi, -sin_phi);
        //Technically vec3 but here vec2, since the third coordinate is 0 and then only use it for dot products
        let world_dir = vec2<f16>(cos_phi, sin_phi);
        let v_dot_w = dot(view_dir.xy, world_dir);
        let projected_normal_len_inv = inverseSqrt(1. - dot(world_dir, n_cross_v) * dot(world_dir, n_cross_v) / (1. - v_dot_w * v_dot_w));
        let sin_n_scaled = dot(normal.xy, world_dir) - v_dot_w * cos_n_scaled;
        let sin_n = sin_n_scaled * projected_normal_len_inv;
        var cos_h1 = -sin_n;
        var cos_h2 = sin_n;
        let cos_0 = get_sample(local_param, origin, position, view_dir, sin_n, noise.y, 0., 0., dir, radius, pix_dif, world_radius_mul);
        cos_h1 = max(cos_0.x, cos_h1);
        cos_h2 = max(cos_0.y, cos_h2);
        let cos_1 = get_sample(local_param, origin, position, view_dir, sin_n, noise.y, 0., 1., dir, radius, pix_dif, world_radius_mul);
        cos_h1 = max(cos_1.x, cos_h1);
        cos_h2 = max(cos_1.y, cos_h2);
        let cos_2 = get_sample(local_param, origin, position, view_dir, sin_n, noise.y, 0., 2., dir, radius, pix_dif, world_radius_mul);
        cos_h1 = max(cos_2.x, cos_h1);
        cos_h2 = max(cos_2.y, cos_h2);
        // fast_sqrt here gives bad precision
        let sin_h1 = sqrt(1. - cos_h1 * cos_h1);
        let sin_h2 = -sqrt(1. - cos_h2 * cos_h2);
        let h1_p_h2_unsigned = fast_acos(cos_h1 * cos_h2 - sin_h1 * sin_h2);
        //let h1_p_h2 = sign(cos_h2 - cos_h1) * h1_p_h2_unsigned;
        let h1_p_h2 = select(-h1_p_h2_unsigned, h1_p_h2_unsigned, cos_h1 < cos_h2);
        // cos(2. * h1p - n) = cos(2. * h1p) cos(n) + sin(2 * h1p) sin(n)
        //                   = (1. - 2 sin(h1p)^2) cos(n) + 2 cos(h1p) sin(h1p) sin(n)
        // sin(h) * sin(h - n)
        let vis_1 = sin_h1 * (sin_h1 * cos_n_scaled - cos_h1 * sin_n_scaled);
        let vis_2 = sin_h2 * (sin_h2 * cos_n_scaled - cos_h2 * sin_n_scaled);
        visibility += 2. * (vis_1 + vis_2 + h1_p_h2 * sin_n_scaled);
    }
    {
        let cos_phi = cos_phi_init * cos_phi_mul - sin_phi_init * sin_phi_mul;
        let sin_phi = sin_phi_init * cos_phi_mul + cos_phi_init * sin_phi_mul;
        let dir = vec2<f16>(cos_phi, -sin_phi);
        let world_dir = vec2<f16>(cos_phi, sin_phi);
        let v_dot_w = dot(view_dir.xy, world_dir);
        let projected_normal_len_inv = inverseSqrt(1. - dot(world_dir, n_cross_v) * dot(world_dir, n_cross_v) / (1. - v_dot_w * v_dot_w));
        let sin_n_scaled = dot(normal.xy, world_dir) - v_dot_w * cos_n_scaled;
        let sin_n = sin_n_scaled * projected_normal_len_inv;
        var cos_h1 = -sin_n;
        var cos_h2 = sin_n;
        let cos_0 = get_sample(local_param, origin, position, view_dir, sin_n, noise.y, 1., 0., dir, radius, pix_dif, world_radius_mul);
        cos_h1 = max(cos_0.x, cos_h1);
        cos_h2 = max(cos_0.y, cos_h2);
        let cos_1 = get_sample(local_param, origin, position, view_dir, sin_n, noise.y, 1., 1., dir, radius, pix_dif, world_radius_mul);
        cos_h1 = max(cos_1.x, cos_h1);
        cos_h2 = max(cos_1.y, cos_h2);
        let cos_2 = get_sample(local_param, origin, position, view_dir, sin_n, noise.y, 1., 2., dir, radius, pix_dif, world_radius_mul);
        cos_h1 = max(cos_2.x, cos_h1);
        cos_h2 = max(cos_2.y, cos_h2);
        // fast_sqrt here gives bad precision
        let sin_h1 = sqrt(1. - cos_h1 * cos_h1);
        let sin_h2 = -sqrt(1. - cos_h2 * cos_h2);
        let h1_p_h2_unsigned = fast_acos(cos_h1 * cos_h2 - sin_h1 * sin_h2);
        //let h1_p_h2 = sign(cos_h2 - cos_h1) * h1_p_h2_unsigned;
        let h1_p_h2 = select(-h1_p_h2_unsigned, h1_p_h2_unsigned, cos_h1 < cos_h2);
        // cos(2. * h1p - n) = cos(2. * h1p) cos(n) + sin(2 * h1p) sin(n)
        //                   = (1. - 2 sin(h1p)^2) cos(n) + 2 cos(h1p) sin(h1p) sin(n)
        // sin(h) * sin(h - n)
        let vis_1 = sin_h1 * (sin_h1 * cos_n_scaled - cos_h1 * sin_n_scaled);
        let vis_2 = sin_h2 * (sin_h2 * cos_n_scaled - cos_h2 * sin_n_scaled);
        visibility += 2. * (vis_1 + vis_2 + h1_p_h2 * sin_n_scaled);
    }
    /*
        {
        let cos_phi = cos_phi_init * cos_2_phi_mul - sin_phi_init * sin_2_phi_mul;
        let sin_phi = sin_phi_init * cos_2_phi_mul + cos_phi_init * sin_2_phi_mul;
        let dir = vec2<f32>(cos_phi, -sin_phi);
        let world_dir = vec3<f32>(cos_phi, sin_phi, 0.);
        let projected_normal_len_inv = inverseSqrt(1. - dot(world_dir, n_cross_v) * dot(world_dir, n_cross_v) / (1. - dot(view_dir, world_dir) * dot(view_dir, world_dir)));
        let sin_n_scaled = dot(normal, world_dir) - dot(view_dir, world_dir) * cos_n_scaled;
        let sin_n = sin_n_scaled * projected_normal_len_inv;
        let cos_n_plus_pi_2 = -sin_n;
        let cos_n_minus_pi_2 = sin_n;
        var cos_h1 = cos_n_plus_pi_2;
        var cos_h2 = cos_n_minus_pi_2;
        let cos_0 = get_sample(origin, position, view_dir, cos_n_plus_pi_2, noise.y, 2., 0., dir, radius, pix_dif, world_radius_mul);
        cos_h1 = max(cos_0.x, cos_h1);
        cos_h2 = max(cos_0.y, cos_h2);
        let cos_1 = get_sample(origin, position, view_dir, cos_n_plus_pi_2, noise.y, 2., 1., dir, radius, pix_dif, world_radius_mul);
        cos_h1 = max(cos_1.x, cos_h1);
        cos_h2 = max(cos_1.y, cos_h2);
        let cos_2 = get_sample(origin, position, view_dir, cos_n_plus_pi_2, noise.y, 2., 2., dir, radius, pix_dif, world_radius_mul);
        cos_h1 = max(cos_2.x, cos_h1);
        cos_h2 = max(cos_2.y, cos_h2);
        let sin_h1 = sqrt(1. - cos_h1 * cos_h1);
        let sin_h2 = - sqrt(1. - cos_h2 * cos_h2);
        let h1_p_h2_unsigned = fast_acos(cos_h1 * cos_h2 - sin_h1 * sin_h2);
        let h1_p_h2 = select(-h1_p_h2_unsigned, h1_p_h2_unsigned, cos_h1 < cos_h2);
        let vis_1 = sin_h1 * (sin_h1 * cos_n_scaled - cos_h1 * sin_n_scaled);
        let vis_2 = sin_h2 * (sin_h2 * cos_n_scaled - cos_h2 * sin_n_scaled);
        visibility += 2. * (vis_1 + vis_2 + h1_p_h2 * sin_n_scaled);
    }*/

    var scaled_visibility = pow(0.25 * visibility / f16(kernel_size), 1.35);
    out.ssao = f32(scaled_visibility);
    return out;
}
