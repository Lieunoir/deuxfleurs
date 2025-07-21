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
var t_d: texture_2d<f32>;
@group(1) @binding(3)
var s: sampler;
@group(1) @binding(4)
var t_v: texture_2d<f32>;

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
    let visibility   = textureLoad(t_v, coords / 2, 0).x;
	let view_dir = normalize(camera.view_pos.xyz - position);

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
