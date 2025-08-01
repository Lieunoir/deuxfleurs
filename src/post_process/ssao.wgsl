struct Parameters {
    frame_offset: u32,
    slices: u32,
    samples: u32,
    _pad3: u32,
    depth_linearize_mul: f32,
    depth_linearize_add: f32,
    _pad4: u32,
    _pad5: u32,
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

const pos = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos[index], 0.0, 1.0);
}

const PI: f32 = 3.14159265359;
const tan_pi_0125 = 0.41421356237;

fn view_from_screen_coord(coord: vec2<f32>, linear_depth_sample: f32) -> vec3<f32> {
    // reconstruct view-space position from the screen coordinate and view space depth.
    return vec3<f32>(
        (vec2<f32>(- 2. * 0.90225565 * tan_pi_0125, 2. * tan_pi_0125) * coord + vec2<f32>(1. * 0.90225565 * tan_pi_0125, - 1. * tan_pi_0125)) * linear_depth_sample,
        linear_depth_sample
    );
}

fn linearize_depth(depth: f32) -> f32 {
    return param.depth_linearize_mul / (param.depth_linearize_add + depth);
}

fn fast_sqrt(x: f32) -> f32 {
    return bitcast<f32>(0x1FBD1DF5 + (bitcast<i32>(x) >> 1));
}

fn fast_acos(x: f32) -> f32 {
    let abs_x = abs(x);
    var res = 1.5707963267948966 - 0.1565827644218014 * abs_x;
    res *= fast_sqrt(1.0 - abs_x);
    return select(PI - res, res, x >= 0);
}

const g : f32= 1.32471795724474602596;
const a1: f32 = 1.0 / g;
const a2: f32 = 1.0 / (g * g);

// mapping each pixel to a hilbert curve index, then taking a value from the Roberts R2 quasirandom sequence for it
fn hilbert_r2_blue_noisef(p: vec2<u32>) -> vec2<f32> {
    var x = textureLoad(hilbert, vec2<u32>(p.x % 64, p.y % 64), 0).x;
    x += param.frame_offset;
    return vec2<f32>(fract(0.5 + vec2<f32>(a1, a2) * f32(x)));
}

fn calculate_edges(centerZ: f32, leftZ: f32, rightZ: f32, topZ: f32, bottomZ: f32) -> vec4<f32> {
    var edgesLRTB = vec4<f32>(leftZ, rightZ, topZ, bottomZ) - centerZ;

    let slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
    let slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
    let edgesLRTBSlopeAdjusted = edgesLRTB + vec4<f32>(slopeLR, -slopeLR, slopeTB, -slopeTB);
    edgesLRTB = min(abs(edgesLRTB), abs(edgesLRTBSlopeAdjusted));
    //remember: centerZ < 0.
    return vec4<f32>(saturate((1.25 + edgesLRTB / (centerZ * 0.011))));
}

// packing/unpacking for edges; 2 bits per edge mean 4 gradient values (0, 0.33, 0.66, 1) for smoother transitions!
fn pack_edges(in: vec4<f32>) -> f32 {
    var edgesLRTB = round(saturate(in) * 2.9);
    return dot(edgesLRTB, vec4<f32>(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0)) ;
}

const sample_factor: f32 = 1.;

struct FragmentOutput {
    @location(0) ssao: f32,
    @location(1) edges: f32,
}
const kernel_size = 2.;
const samples_per_slice = 2.;
const phi_mul = PI / kernel_size;
const vis_scale = 0.25 / kernel_size;
const max_radius_pix = 64;
override max_mip_level: f32 = 0;

@fragment
fn fs_main(@builtin(position) fcoords: vec4<f32>) -> FragmentOutput {
    var out: FragmentOutput;
    let buffer_size = vec2<f32>(textureDimensions(t_d));
    let pix_dif = vec2<f32>(1.) / buffer_size;
    let coords = vec2<u32>(floor(fcoords.xy * sample_factor));
    let noise = hilbert_r2_blue_noisef(coords);
    let origin = sample_factor * fcoords.xy * pix_dif;
    let gather_offset = - vec2<f32>(0.25) * pix_dif;
    let values_ul = textureGather(0, t_d, s, origin + gather_offset);
    let values_br = textureGather(0, t_d, s, origin + gather_offset, vec2<i32>(1, 1));
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
    out.edges = packed_edges;

    let position = view_from_screen_coord(origin, depth);
    let normal_sample = textureLoad(t_n, coords, 0).xyz;
    let normal = normalize(normal_sample * 2. - vec3<f32>(1.));
    let view_dir = normalize(-position);
    let n_cross_v = cross(normal, view_dir);
    let cos_n_scaled = dot(view_dir, normal);

    //if abs(normal_sample[0]) + abs(normal_sample[1]) + abs(normal_sample[2]) < 0.01 {
	//    discard;
    //}

    // GTAO
    var visibility = 0.0;
    //let kernel_size: u32 = param.slices;
    //let samples_per_slice = param.samples;
    let world_distance = linearize_depth(1.);
    let world_radius_mul = 10. / world_distance;
    //let world_radius = 0.02 * world_distance;
    let wanted_screen_radius = 2. * world_distance / depth;
    let radius = max_radius_pix * min(wanted_screen_radius, 1.);
    for (var i = 0.; i < kernel_size; i += 1.) {
        let phi = (noise.x + i) * phi_mul;
        let cos_phi = cos(phi);
        let sin_phi = sin(phi);
        //let sin_phi = fast_sqrt(1. - cos_phi * cos_phi);
        let dir = vec2<f32>(cos_phi, -sin_phi);
        let world_dir = vec3<f32>(cos_phi, sin_phi, 0.);
        let projected_normal_len_inv = inverseSqrt(1. - pow(dot(world_dir, n_cross_v), 2.) / (1. - pow(dot(view_dir, world_dir), 2.)));
        let sin_n_scaled = dot(normal, world_dir) - dot(view_dir, world_dir) * cos_n_scaled;
        let sin_n = sin_n_scaled * projected_normal_len_inv;
        let cos_n_plus_pi_2 = -sin_n;
        let cos_n_minus_pi_2 = sin_n;
        var cos_h1 = 0.;
        var cos_h2 = 0.;

        for (var j = 0.; j < samples_per_slice; j += 1.) {
            let step_noise = fract(noise.y + (i + j * samples_per_slice) * 0.6180339887498948482f);
            let sample_offset_length = pow((step_noise + j) / samples_per_slice, 2.) * radius;
            // compile time version, so mip_level is known at compile time too
            let sample_offset_length_approx = pow((0.5 + j) / samples_per_slice, 2.) * max_radius_pix;
            let sample_offset = sample_offset_length * pix_dif * dir;
            //let mip_level = clamp(log2(sample_offset_length_approx) - 3.3, 0., max_mip_level);
            //let mip_level = max_mip_level;
            let mip_level = 0.;

            let sample_coords_1 = vec2<f32>(origin + sample_offset);
            let sample_coords_2 = vec2<f32>(origin - sample_offset);
            let sample_depth_1 = textureSampleLevel(t_d, s, sample_coords_1, mip_level).x;
            let sample_depth_2 = textureSampleLevel(t_d, s, sample_coords_2, mip_level).x;

            let sample_1 = view_from_screen_coord(sample_coords_1, sample_depth_1);
            let sample_2 = view_from_screen_coord(sample_coords_2, sample_depth_2);
            let dir_1 = sample_1 - position;
            let dir_2 = sample_2 - position;

            let d_s_1_norm_inv = inverseSqrt(dot(dir_1, dir_1));
            let d_s_2_norm_inv = inverseSqrt(dot(dir_2, dir_2));

            let d_s_1 = dir_1 * d_s_1_norm_inv;
            let d_s_2 = dir_2 * d_s_2_norm_inv;

            // (d - 0.02 * world_distance) * 5. / (0.02 * world_distance)
            // = (d / (0.02 * world_distance) - 1.) * 5.)
            //let l_s_1 = saturate((d_s_1_norm - world_radius) * 5. / world_radius);
            //let l_s_2 = saturate((d_s_2_norm - world_radius) * 5. / world_radius);
            let l_s_1 = saturate(world_radius_mul / d_s_1_norm_inv - 5.);
            let l_s_2 = saturate(world_radius_mul / d_s_2_norm_inv - 5.);
            let dot_s_1 = mix(dot(d_s_1, normal), cos_n_plus_pi_2, l_s_1);
            let dot_s_2 = mix(dot(d_s_2, normal), cos_n_minus_pi_2, l_s_2);
            //let dot_s_1 = dot(d_s_1, view_dir);
            //let dot_s_2 = dot(d_s_2, view_dir);
            cos_h1 = max(cos_h1, dot_s_1);
            cos_h2 = max(cos_h2, dot_s_2);
        }

        //let h1p = fast_acos(cos_h1);
        //let h2p = -fast_acos(cos_h2);
        ////let h1p = n + max(h1 - n, -PI * 0.5);
        ////let h2p = n + min(h2 - n, PI * 0.5);
        //let vis_1 = - cos(2. * h1p - n) + 2. * h1p * sin_n;
        //let vis_2 = - cos(2. * h2p - n) + 2. * h2p * sin_n;
        //local_visibility += (vis_1 + vis_2) / projected_normal_len_inv;
        // fast_sqrt here gives bad precision
        let sin_h1 = sqrt(1. - cos_h1 * cos_h1);
        let sin_h2 = - sqrt(1. - cos_h2 * cos_h2);
        let h1_p_h2_unsigned = fast_acos(cos_h1 * cos_h2 - sin_h1 * sin_h2);
        let h1_p_h2 = select(-h1_p_h2_unsigned, h1_p_h2_unsigned, cos_h1 < cos_h2);
        // cos(2. * h1p - n) = cos(2. * h1p) cos(n) + sin(2 * h1p) sin(n)
        //                   = (1. - 2 sin(h1p)^2) cos(n) + 2 cos(h1p) sin(h1p) sin(n)
        // sin(h) * sin(h - n)
        let vis_1 = sin_h1 * (sin_h1 * cos_n_scaled - cos_h1 * sin_n_scaled);
        let vis_2 = sin_h2 * (sin_h2 * cos_n_scaled - cos_h2 * sin_n_scaled);
        visibility += 2. * (vis_1 + vis_2 + h1_p_h2 * sin_n_scaled);
    }

    var scaled_visibility = pow(0.25 * visibility / f32(kernel_size), 1.35);
    out.ssao = scaled_visibility;
    return out;
}
