pub const PBR_FUN: &str = "
const PI: f32 = 3.14159;

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
";

// Requires :
// light.position
// pos
// camera.view_pos
// normal
// lambertian
pub const PBR_FRAG: &str = "
	let light_dir = normalize(light.position - pos);
	let view_dir = normalize(camera.view_pos.xyz - pos);
	let half_dir = normalize(view_dir + light_dir);
	let F0 = vec3<f32>(0.04, 0.04, 0.04);
	let D = DistributionGGX(normal, half_dir, 0.15);
	let F = fresnelSchlick(dot(half_dir, normal), F0);
	let G = GeometrySmith(normal, view_dir, light_dir, 0.01);
	let f_ct = D * F * G / (4. * dot(view_dir, normal) * dot(light_dir, normal));
	let kd = 1.0;
	let result = (kd * lambertian + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);
";

pub const COLORMAP_UNIFORM: &str = "
struct ColorMapUniform {
    k_red_vec1: vec4<f32>,
    k_red_vec2: vec4<f32>,
    k_green_vec1: vec4<f32>,
    k_green_vec2: vec4<f32>,
    k_blue_vec1: vec4<f32>,
    k_blue_vec2: vec4<f32>,
}

@group(2) @binding(0)
var<uniform> colormap_uniform: ColorMapUniform;

//https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
fn colormap(x: f32) -> vec3<f32> {
    let v4: vec4<f32> = vec4<f32>(1.0, x, x*x, x*x*x);
    let v2: vec4<f32> = v4 * v4.w * x;
    //let v2: vec2<f32> = vec2<f32>(0., 0.);
    return vec3<f32>(
        dot(v4, colormap_uniform.k_red_vec1) + dot(v2, colormap_uniform.k_red_vec2),
        dot(v4, colormap_uniform.k_green_vec1) + dot(v2, colormap_uniform.k_green_vec2),
        dot(v4, colormap_uniform.k_blue_vec1) + dot(v2, colormap_uniform.k_blue_vec2),
    );
}
";
