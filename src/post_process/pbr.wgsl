struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

@group(1) @binding(0)
var t_a: texture_2d<f32>;
@group(1) @binding(1)
var t_n: texture_2d<f32>;
@group(1) @binding(2)
//var t_d: texture_depth_2d;
var t_d: texture_2d<f32>;
@group(1) @binding(3)
var s: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

const pos = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index : u32,
    ) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos[index], 0.0, 1.0);
    return out;
}

const PI: f32 = 3.14159265359;

// PBR functions taken from https://learnopengl.com/PBR/Theory
fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, a: f32) -> f32 {
    let a2     = a*a;
    let NdotH  = max(dot(N, H), 0.0);
    let NdotH2 = NdotH*NdotH;

    let nom    = a2;
    var denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    denom        = PI * denom * denom;

    return nom / denom;
}

fn GeometrySchlickGGX(NdotV: f32, k: f32) -> f32
{
    let nom   = NdotV;
    let denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, k: f32) -> f32
{
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx1 = GeometrySchlickGGX(NdotV, k);
    let ggx2 = GeometrySchlickGGX(NdotL, k);

    return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32>
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

fn world_from_screen_coord(coord : vec2<f32>, depth_sample: f32) -> vec3<f32> {
    // reconstruct world-space position from the screen coordinate.
    let posClip = vec4(coord.x * 2.0 - 1.0, 1.0 - 2.0 * coord.y, depth_sample, 1.0);
    let posWorldW = camera.view_inv * posClip;
    let posWorld = posWorldW.xyz / posWorldW.www;
    return posWorld;
}

fn pcg3d(v_orig: vec3<u32>) -> vec3<u32> {
    var v = v_orig * 1664525 + 1013904223;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    v.x ^= v.x>>16u;
    v.y ^= v.y>>16u;
    v.z ^= v.z>>16u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    return v;
}

fn fast_sqrt(x: f32) -> f32 {
    return f32(0x1FBD1DF5 + (u32(x) >> 1));
}

fn fast_acos(x: f32) -> f32 {
    var res = -0.156583 * abs(x) + PI / 2.0;
    res *= fast_sqrt(1. - abs(x));
    return select(PI - res, res, x >= 0);
}

const randoms = array<f32, 64>(0.9073287956637583, 0.8953753268762352, 0.3220086438462023, 0.007605212815564366, 0.01591998320496857, 0.16333876403470682, 0.7633080275109663, 0.6253689714158442, 0.9796289477520932, 0.47768855334816007, 0.20994347509627442, 0.42647190872472107, 0.3264460758651072, 0.603054743243745, 0.4421765326581557, 0.13635578498504275,
    0.5480187485794791, 0.7002945901365113, 0.04093307934142931, 0.8409299478779066, 0.3657819008493858, 0.3872717431211139, 0.5296179826887955, 0.3549791699992324, 0.03845149501235379, 0.9752711547848418, 0.20037853481683254, 0.31096408522103347, 0.9594224215818684, 0.9629871955616451, 0.4983265536276734, 0.002695323442428843,
    0.35680469302547124, 0.6338448300380964, 0.26924514548124223, 0.5489805045735846, 0.38712840331458065, 0.34813314754718905, 0.21110995223799223, 0.06735202851625521, 0.22925362499197766, 0.9693096630885775, 0.13104928603132715, 0.5136988570398621, 0.993335107309559, 0.8645336635925384, 0.05809545593417287, 0.12120304216110633,
    0.22041811198640138, 0.17310442191243958, 0.26970976141108405, 0.7577908143740093, 0.3530547214528106, 0.7158705393016846, 0.4373999583878948, 0.8503007357829833, 0.06923972709448556, 0.7685377089983041, 0.2800583414822193, 0.4926678074779679, 0.8794457785989035, 0.22453667177222958, 0.5565299827383392, 0.6752055012992703);

@fragment
fn fs_main(@builtin(position) fcoords : vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(floor(fcoords.xy));
    let albedo   = textureLoad(t_a, coords, 0);
    if(albedo.w < 0.01) {
        discard;
    }
    let buffer_size = textureDimensions(t_d);
    let depth = textureSample(t_d, s, fcoords.xy / vec2<f32>(buffer_size)).x;
    let position = world_from_screen_coord(fcoords.xy / vec2<f32>(buffer_size), depth);
    let normal   = normalize(textureLoad(t_n, coords, 0).xyz * 2. - vec3<f32>(1.));
	let view_dir = normalize(camera.view_pos.xyz - position);

    // SSAO
    //let rand = pcg3d(vec3<u32>(bitcast<u32>(coords.x), bitcast<u32>(coords.y), bitcast<u32>(depth)));
    //let randoms_i_1 = rand.x & 63;
    //let randoms_i_2 = rand.y & 63;
    //let randoms_i_3 = rand.z & 63;
    //let random_vec = vec3<f32>(
    //    2. * randoms[randoms_i_1] - 1.,
    //    2. * randoms[randoms_i_2] - 1.,
    //    2. * randoms[randoms_i_3] - 1.,
    //);
    //let tangent = normalize(random_vec - normal * dot(random_vec, normal));
    //let bitangent = cross(normal, tangent);
    //let TBN = mat3x3<f32>(tangent, bitangent, normal);
    //var occlusion = 0.0;
    //let kernelSize: u32 = 6u;
    //let radius = 0.05;
    //for(var i: u32 = 0; i < kernelSize; i+=1)
    //{
    //    let rand2 = pcg3d(vec3<u32>(i, bitcast<u32>(depth), bitcast<u32>(coords.x * coords.y)));
    //    let angle_bias = 0.2;
    //    //let theta = (angle_bias + randoms[rand2.x & 63] * (1. - angle_bias)) * PI;
    //    let theta = (randoms[rand2.x & 63] - 0.5) * PI;
    //    let phi = randoms[rand2.y & 63] * PI;
    //    let i4 = rand2.z & 63;
    //    let offset = TBN * vec3<f32>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)); // from tangent to world
    //    let sample_pos = position + offset * radius * pow(randoms[i4], 2.);

    //    var sample = vec4(sample_pos, 1.0);
    //    sample = camera.view_proj * sample;    // from world to clip-space
    //    //let bias = 0.0025;
    //    let bias = 0.0000001;
    //    sample /= sample.w;               // perspective divide
    //    sample.x = sample.x * 0.5 + 0.5;
    //    sample.y = 0.5 - sample.y * 0.5;
    //    let sample_depth = sample.z;
    //    let origin = fcoords.xy / vec2<f32>(buffer_size);
    //    let sample_off = origin - sample.xy;
    //    let sampleDepth_1 = textureSample(t_d, s, origin + sample_off * 0.3).x;
    //    let sampleDepth_2 = textureSample(t_d, s, origin + sample_off * 0.6).x;
    //    let sampleDepth_3 = textureSample(t_d, s, sample.xy).x;
    //    let sample_world_pos_1 = world_from_screen_coord(origin + sample_off * 0.3, sampleDepth_1);
    //    let sample_world_pos_2 = world_from_screen_coord(origin + sample_off * 0.6, sampleDepth_2);
    //    let sample_world_pos_3 = world_from_screen_coord(sample.xy, sampleDepth_3);
    //    let range_check_1 = smoothstep(0.0, 1.0, radius / abs(sample_world_pos_1.z - position.z));
    //    let range_check_2 = smoothstep(0.0, 1.0, radius / abs(sample_world_pos_2.z - position.z));
    //    let range_check_3 = smoothstep(0.0, 1.0, radius / abs(sample_world_pos_3.z - position.z));
    //    let occlusion_1 = select(0., 1. / f32(kernelSize), sampleDepth_1 + bias < sample_depth) * range_check_1;
    //    let occlusion_2 = max(occlusion_1, select(0., 1. / f32(kernelSize), sampleDepth_2 + bias < sample_depth) * range_check_2);
    //    occlusion += max(occlusion_2, select(0., 1. / f32(kernelSize), sampleDepth_3 + bias < sample_depth) * range_check_3);
    //}

    // GTAO
    var visibility = 0.0;
    let kernelSize: u32 = 3u;
    let origin = fcoords.xy / vec2<f32>(buffer_size);
    let pix_dif = vec2<f32>(1.) / vec2<f32>(buffer_size);
    let camera_distance = sqrt(dot(camera.view_pos.xyz - position,camera.view_pos.xyz - position));
    let wanted_radius = 20. / camera_distance * pix_dif;
    let radius = vec2<f32>(
        min(wanted_radius.x, 20. * pix_dif.x),
        min(wanted_radius.y, 20. * pix_dif.y),
    );
    //let radius = min(min(0.005 / camera_distance, 30. * pix_dif.x), 30. * pix_dif.y);
    for(var i: u32 = 0; i < kernelSize; i+=1) {
        let rand = pcg3d(vec3<u32>(i, bitcast<u32>(depth), bitcast<u32>(coords.x * coords.y)));
        let phi = (randoms[rand.x & 63] + f32(i)) * PI / f32(kernelSize);
        let dir = vec2<f32>(cos(phi), -sin(phi)) * radius * randoms[rand.y & 63];
        //let dir = vec2<f32>(cos(phi), -sin(phi)) * radius;
        let world_dir = normalize(world_from_screen_coord(origin + dir, 0.) - position);
        let ortho_direction_v = world_dir - dot(world_dir, view_dir) * view_dir;
        let slice_plane_normal = normalize(cross(world_dir, view_dir));
        let projected_normal = normal - dot(normal, slice_plane_normal) * slice_plane_normal;
        let sign_n = sign(dot(ortho_direction_v, projected_normal));
        let cos_n = saturate(dot(view_dir,normalize(projected_normal)));
        let n = sign_n * acos(cos_n);


        let sample_coords_1 =  vec2<f32>( origin + 0.33 * dir);
        let sample_coords_2 =  vec2<f32>( origin + 0.66 * dir);
        let sample_coords_3 =  vec2<f32>( origin + 1.00 * dir);
        let sample_coords_1p =  vec2<f32>(origin - 0.33 * dir);
        let sample_coords_2p =  vec2<f32>(origin - 0.66 * dir);
        let sample_coords_3p =  vec2<f32>(origin - 1.00 * dir);
        let sample_depth_1 = textureSample(t_d, s,  sample_coords_1).x;
        let sample_depth_2 = textureSample(t_d, s,  sample_coords_2).x;
        let sample_depth_3 = textureSample(t_d, s,  sample_coords_3).x;
        let sample_depth_1p = textureSample(t_d, s, sample_coords_1p).x;
        let sample_depth_2p = textureSample(t_d, s, sample_coords_2p).x;
        let sample_depth_3p = textureSample(t_d, s, sample_coords_3p).x;

        let sample_1 =  world_from_screen_coord(sample_coords_1, sample_depth_1);
        let sample_2 =  world_from_screen_coord(sample_coords_2, sample_depth_2);
        let sample_3 =  world_from_screen_coord(sample_coords_3, sample_depth_3);
        let sample_1p = world_from_screen_coord(sample_coords_1p, sample_depth_1p);
        let sample_2p = world_from_screen_coord(sample_coords_2p, sample_depth_2p);
        let sample_3p = world_from_screen_coord(sample_coords_3p, sample_depth_3p);

        let d_s_1_squared = dot(sample_1  - position, sample_1 - position);
        let d_s_2_squared = dot(sample_2  - position, sample_2 - position);
        let d_s_3_squared = dot(sample_3  - position, sample_3 - position);
        let d_t_1_squared = dot(sample_1p - position, sample_1p - position);
        let d_t_2_squared = dot(sample_2p - position, sample_2p - position);
        let d_t_3_squared = dot(sample_3p - position, sample_3p - position);

        let d_s_1 = normalize(sample_1 - position);
        let d_s_2 = normalize(sample_2 - position);
        let d_s_3 = normalize(sample_3 - position);
        let d_t_1 = normalize(sample_1p - position);
        let d_t_2 = normalize(sample_2p - position);
        let d_t_3 = normalize(sample_3p - position);

        let world_radius = 10000.;
        //let world_radius = 0.5;
        let l_s_1 = saturate((sqrt(d_s_1_squared) - world_radius) / world_radius);
        let l_s_2 = saturate((sqrt(d_s_2_squared) - world_radius) / world_radius);
        let l_s_3 = saturate((sqrt(d_s_3_squared) - world_radius) / world_radius);
        let l_t_1 = saturate((sqrt(d_t_1_squared) - world_radius) / world_radius);
        let l_t_2 = saturate((sqrt(d_t_2_squared) - world_radius) / world_radius);
        let l_t_3 = saturate((sqrt(d_t_3_squared) - world_radius) / world_radius);
        let dot_s_1 = (1. - l_s_1) * dot(d_s_1, view_dir) + l_s_1 * cos(n + PI * 0.5);
        let dot_s_2 = (1. - l_s_2) * dot(d_s_2, view_dir) + l_s_2 * cos(n + PI * 0.5);
        let dot_s_3 = (1. - l_s_3) * dot(d_s_3, view_dir) + l_s_3 * cos(n + PI * 0.5);
        let dot_t_1 = (1. - l_t_1) * dot(d_t_1, view_dir) + l_t_1 * cos(n - PI * 0.5);
        let dot_t_2 = (1. - l_t_2) * dot(d_t_2, view_dir) + l_t_2 * cos(n - PI * 0.5);
        let dot_t_3 = (1. - l_t_3) * dot(d_t_3, view_dir) + l_t_3 * cos(n - PI * 0.5);

        let cos_h1 = max(max(dot_s_1, dot_s_2), dot_s_3);
        let cos_h2 = max(max(dot_t_1, dot_t_2), dot_t_3);
        let h1 = acos(cos_h1);
        let h2 = -acos(cos_h2);
        let h1p = n + clamp(h1 - n, -PI * 0.5, PI * 0.5);
        let h2p = n + clamp(h2 - n, -PI * 0.5, PI * 0.5);

        let projected_normal_length = sqrt(dot(projected_normal, projected_normal));
        let projected_normal_length_2 = 0.05 * projected_normal_length + (1. - 0.05);

        let local_visibility = 0.25 * projected_normal_length * (
            - cos(2. * h1p - n) + 2. * cos_n + 2. * (h1p + h2p) * sin(n)
            - cos(2. * h2p - n)
        );
        visibility += local_visibility / f32(kernelSize);
    }
    //visibility = 1.;

    let F0 = vec3<f32>(0.04, 0.04, 0.04);
    let kd = 1.;

	let up =  normalize(vec3<f32>(
	    camera.view_proj[0].y,
	    camera.view_proj[1].y,
		camera.view_proj[2].y
	));
	let right =  normalize(vec3<f32>(
	    camera.view_proj[0].x,
	    camera.view_proj[1].x,
		camera.view_proj[2].x
	));
	let forward =  normalize(vec3<f32>(
	    camera.view_proj[0].z,
	    camera.view_proj[1].z,
		camera.view_proj[2].z
	));
	let light_dir = normalize(right - up - forward);
	//let light_dir = normalize(light.position - position);
	let half_dir = normalize(view_dir + light_dir);
	let D = DistributionGGX(normal, half_dir, albedo.w);
	let F = fresnelSchlick(dot(half_dir, normal), F0);
	let G = GeometrySmith(normal, view_dir, light_dir, albedo.w);
	let f_ct = D * F * G / (4. * dot(view_dir, normal) * dot(light_dir, normal));
	//var result = 0.55 * (kd * albedo.xyz * visibility + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);
	var result = 0.5 * 0.55 * (kd * albedo.xyz + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);

	let light_dir_2 = normalize(-right + up - forward);
	//let light_dir_2 = normalize(vec3<f32>(1., 1., -1.));
	let half_dir_2 = normalize(view_dir + light_dir_2);
	let D2 = DistributionGGX(normal, half_dir_2, albedo.w);
	let F2 = fresnelSchlick(dot(half_dir_2, normal), F0);
	let G2 = GeometrySmith(normal, view_dir, light_dir_2, albedo.w);
	let f_ct_2 = D2 * F2 * G2 / (4. * dot(view_dir, normal) * dot(light_dir_2, normal));
	//result += 1.6 * (kd * albedo.xyz * visibility + PI * f_ct_2) * light.color * max(dot(normal, light_dir_2), 0.0);
	result += 0.5 * 1.6 * (kd * albedo.xyz + PI * f_ct_2) * light.color * max(dot(normal, light_dir_2), 0.0);

	let light_dir_3 = normalize(right + up + forward);
	let half_dir_3 = normalize(view_dir + light_dir_3);
	let D3 = DistributionGGX(normal, half_dir_3, albedo.w);
	let F3 = fresnelSchlick(dot(half_dir_3, normal), F0);
	let G3 = GeometrySmith(normal, view_dir, light_dir_3, albedo.w);
	let f_ct_3 = D3 * F3 * G3 / (4. * dot(view_dir, normal) * dot(light_dir_3, normal));
	//result += 1.4 * (kd * albedo.xyz * visibility + PI * f_ct_2) * light.color * max(dot(normal, light_dir_3), 0.0);
	result += 0.5 * 1.4 * (kd * albedo.xyz + PI * f_ct_2) * light.color * max(dot(normal, light_dir_3), 0.0);

	//result += 1.2 * kd * albedo.xyz * visibility ;
	//result *= 0.5;
	result *= 2. * visibility;

	//Tone mapping
	let m1 = mat3x3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777,
    );
    let m2 = mat3x3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602,
    );
    let v = m1 * result;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return vec4<f32>(clamp(m2 * (a / b), vec3(0.0), vec3(1.0)), 1.0);
    //return vec4<f32>(result, 1.0);
}
