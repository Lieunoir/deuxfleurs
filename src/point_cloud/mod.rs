use crate::texture;
use crate::util;
use wgpu::util::DeviceExt;
use std::collections::HashMap;

pub trait DataBufferBuilder {
    fn build_data_buffer(&self, device: &wgpu::Device, positions: &[[f32; 3]]) -> wgpu::Buffer;
}

impl<T> DataBufferBuilder for Option<&T>
where
    T: DataBufferBuilder,
{
    fn build_data_buffer(&self, device: &wgpu::Device, positions: &[[f32; 3]]) -> wgpu::Buffer {
        match self {
            Some(builder) => builder.build_data_buffer(device, positions),
            None => {
                let mut gpu_vertices = Vec::with_capacity(positions.len());
                for position in positions.iter() {
                    let vertex = PointData {
                        position: *position,
                        color: [0., 0., 0.],
                    };
                    gpu_vertices.push(vertex);
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("PC Data Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
        }
    }
}
pub struct PCData {
    scalar: Vec<f32>,
}

impl DataBufferBuilder for PCData {
    fn build_data_buffer(&self, device: &wgpu::Device, positions: &[[f32; 3]]) -> wgpu::Buffer {

        let mut min_d = self.scalar[0];
        let mut max_d = self.scalar[0];
        for data in &self.scalar {
            if *data > max_d {
                max_d = *data;
            }
            if *data < min_d {
                min_d = *data;
            }
        }

        let mut gpu_vertices = Vec::with_capacity(positions.len());
        //for ((position, color), distance) in positions.iter().zip(colors).zip(distance) {
        for (position, data) in positions.iter().zip(self.scalar.iter()) {
                let mut color = [0.; 3];
                let t = (data - min_d) / (max_d - min_d);
                color[0] = t * t;
                color[1] = 2. * t * (1. - t);
                color[2] = (1. - t) * (1. - t);
            let vertex = PointData {
                position: *position,
                color,
            };
            gpu_vertices.push(vertex);
        }
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC Data Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }
}

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PCSettings {
    pub radius: f32,
    pub _padding: [u32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PointVertex {
    position: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PointData {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex for PointVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<PointVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

impl Vertex for PointData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<PointData>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub struct PointCloud {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub num_elements: u32,

    pub datas: HashMap<String, PCData>,
    pub shown_data: Option<String>,
    pub data_to_show: Option<Option<String>>,
    pub settings: PCSettings,
    pub property_changed: bool,
    pub uniform_changed: bool,
    pub show: bool,
    //renderer
    //picker
    //
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    data_buffer: wgpu::Buffer,
    settings_bind_group: wgpu::BindGroup,
    settings_buffer: wgpu::Buffer,
    pipeline_layout: wgpu::PipelineLayout,
}

impl PointCloud {
    fn build_vertex_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        let s2 = 2_f32.sqrt();
        let positions = [
            [-s2, -s2, 0.],
            [s2, -s2, 0.],
            [-s2, s2, 0.],
            [s2, -s2, 0.],
            [s2, s2, 0.],
            [-s2, s2, 0.],
        ];
        let vertices = positions.map(|position| PointVertex { position });
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    fn build_data_buffer(
        device: &wgpu::Device,
        positions: &Vec<[f32; 3]>,
        colors: &Vec<[f32; 3]>,
        //distance: &Vec<f32>,
    ) -> wgpu::Buffer {
        let mut gpu_vertices = Vec::with_capacity(positions.len());
        //for ((position, color), distance) in positions.iter().zip(colors).zip(distance) {
        for (position, color) in positions.iter().zip(colors) {
            let vertex = PointData {
                position: *position,
                color: *color,
            };
            gpu_vertices.push(vertex);
        }
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC Data Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    pub fn new(
        name: String,
        positions: Vec<[f32; 3]>,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        //transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        ) -> Self {
        let settings = PCSettings{
            radius: 0.01,
            _padding: [0; 3],
            color: [0.2, 0.2, 0.8, 1.],
        };
        let settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC settings buffer"),
            contents: bytemuck::cast_slice(&[settings]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let settings_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("pc_settings_bind_group_layout"),
            });
        let settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &settings_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: settings_buffer.as_entire_binding(),
            }],
            label: Some("pc_settings_bind_group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Point cloud Render Pipeline Layout"),
            bind_group_layouts: &[
                camera_light_bind_group_layout,
                //transform_bind_group_layout,
                &settings_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("point cloud shader"),
            //TODO change shader
            source: wgpu::ShaderSource::Wgsl(get_shader(false).into()),
        };
        let render_pipeline = util::create_render_pipeline(
            device,
            &pipeline_layout,
            color_format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[PointVertex::desc(), PointData::desc()],
            shader,
            Some("point cloud field render"),
        );

        let vertex_buffer = Self::build_vertex_buffer(device);
        let colors = vec![[0., 0., 0.]; positions.len()];
        let data_buffer = Self::build_data_buffer(device, &positions, &colors);
        Self {
            name,
            num_elements: positions.len() as u32,
            positions,
            property_changed: false,
            uniform_changed: false,
            datas: HashMap::new(),
            shown_data: None,
            data_to_show: None,
            settings,
            show: true,
            data_buffer,
            render_pipeline,
            settings_bind_group,
            settings_buffer,
            vertex_buffer,
            pipeline_layout,
        }
    }

    pub fn set_radius(&mut self, radius: f32) {
        self.settings.radius = radius;
        self.uniform_changed = true;
    }

    pub fn set_color(&mut self, color: [f32; 4]) {
        self.settings.color = color;
        self.uniform_changed = true;
    }

    pub fn add_scalar(&mut self, name: String, datas: Vec<f32>) -> &mut Self {
        assert!(datas.len() == self.positions.len());
        if let Some(data_name) = &mut self.shown_data {
            if *data_name == name {
                self.data_to_show = Some(Some(data_name.clone()));
                self.shown_data = None;
            }
        }
        self.datas
            .insert(name, PCData { scalar: datas });
        self
    }

    pub fn set_data(&mut self, name: Option<String>) -> &mut Self {
        self.data_to_show = Some(name);
        self
    }

    pub fn refresh_data(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> bool {
        let mut refresh_screen = false;
        if let Some(data) = self.data_to_show.clone() {
            self.data_to_show = None;
            self.shown_data = data;
            let cloud_data = match &self.shown_data {
                Some(data) => self.datas.get(data),
                None => None,
            };
            self.data_buffer = cloud_data.build_data_buffer(device, &self.positions);
            self.property_changed = true;
            refresh_screen = true;
        }
        if self.property_changed {
            let cloud_data = match &self.shown_data {
                Some(data) => self.datas.get(data),
                None => None,
            };

            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("point cloud shader"),
                //TODO change shader
                source: wgpu::ShaderSource::Wgsl(get_shader(cloud_data.is_some()).into()),
            };
            self.render_pipeline = util::create_render_pipeline(
                device,
                &self.pipeline_layout,
                color_format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[PointVertex::desc(), PointData::desc()],
                shader,
                Some("point cloud field render"),
            );
            self.property_changed = false;
            self.uniform_changed = false;
            true
        } else if self.uniform_changed {
            queue.write_buffer(
                &self.settings_buffer,
                0,
                bytemuck::cast_slice(&[self.settings]),
            );
            self.uniform_changed = false;
            true
        } else {
            refresh_screen
        }
    }

    pub fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.show {
            render_pass.set_bind_group(1, &self.settings_bind_group, &[]);
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.data_buffer.slice(..));
            render_pass.draw(0..6, 0..(self.positions.len() as u32));
        }
    }

    pub fn update(&mut self, queue: &mut wgpu::Queue) {
        if self.property_changed {
            queue.write_buffer(
                &self.settings_buffer,
                0,
                bytemuck::cast_slice(&[self.settings]),
            );
            self.property_changed = false;
        }
    }
}

fn get_shader(show_data: bool) -> &'static str {
    if !show_data { SPHERE_SHADER } else { SPHERE_DATA_SHADER }
}

//macro_rules! SHADER { () => {"
const SPHERE_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

//struct TransformUniform {
//    model: mat4x4<f32>,
//    normal: mat4x4<f32>,
//}

struct SettingsUniform {
    radius: f32,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

//@group(1) @binding(0)
//var<uniform> transform: TransformUniform;
@group(1) @binding(0)
var<uniform> settings: SettingsUniform;
struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct DataInput {
    @location(1) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
	@location(0) world_pos: vec3<f32>,
	@location(1) center: vec3<f32>,
	//@location(2) arrow: vec3<f32>,
	//@location(3) radius: f32,
};

@vertex
fn vs_main(
    model: VertexInput,
    data: DataInput,
) -> VertexOutput {
    //let model_matrix = transform.model;
    //let normal_matrix = transform.normal;

    //let world_vector_pos = (model_matrix * vec4<f32>(vector_i.orig_position, 1.)).xyz;
    //// Do we want to scale a vector field if we scale its attached mesh?
    //let world_vector_arrow_t = (model_matrix * vec4<f32>(vector_i.orig_position + vector_i.arrow, 1.)).xyz - world_vector_pos;
    //let arrow_ampl = length(world_vector_arrow_t);
    //let world_vector_arrow = normalize(world_vector_arrow_t);

    //// We define the output we want to send over to frag shader
    var out: VertexOutput;

    let camera_right = normalize(vec3<f32>(camera.view_proj.x.x, camera.view_proj.y.x, camera.view_proj.z.x));
    let camera_up = normalize(vec3<f32>(camera.view_proj.x.y, camera.view_proj.y.y, camera.view_proj.z.y));
    let world_position = data.position + (model.position.x * camera_right + model.position.y * camera_up) * settings.radius;
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.world_pos = world_position;
    out.center = data.position;
    return out;
}

// function from :
// https://iquilezles.org/articles/intersectors/
fn dot2(v: vec3<f32>) -> f32 { return dot(v, v); }


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

fn sphIntersect( ro: vec3<f32>, rd: vec3<f32>, ce: vec3<f32>, ra: f32 ) -> vec2<f32>
{
    let oc = ro - ce;
    let b = dot( oc, rd );
    let c = dot( oc, oc ) - ra*ra;
    var h = b*b - c;
    if( h<0.0 ) { return vec2<f32>(-1.0); } // no intersection
    h = sqrt( h );
    return vec2<f32>( -b-h, -b+h );
}

struct FragOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    let ro = camera.view_pos.xyz;
	let rd = normalize(in.world_pos - camera.view_pos.xyz);
    let ce = in.center;
    let r = settings.radius;
    //let pa = in.orig_position;
    //let pb1 = in.orig_position + 0.5 * in.arrow * 0.1;
    //let pb2 = in.orig_position + in.arrow * 0.1;

    var out: FragOutput;

    let t = sphIntersect( ro, rd, ce, r);
    if(t.x < 0.0) {
        discard;
    }
	let pos = ro + t.x * rd;
	let normal = normalize(pos - ce);

	let light_dir = normalize(light.position - pos);
	let view_dir = normalize(camera.view_pos.xyz - pos);
	let half_dir = normalize(view_dir + light_dir);
	let F0 = vec3<f32>(0.04, 0.04, 0.04);
	let D = DistributionGGX(normal, half_dir, 0.15);
	let F = fresnelSchlick(dot(half_dir, normal), F0);
	let G = GeometrySmith(normal, view_dir, light_dir, 0.01);
	let f_ct = D * F * G / (4. * dot(view_dir, normal) * dot(light_dir, normal));
	let kd = 1.0;
	let lambertian = settings.color;
	let result = (kd * lambertian + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);
	out.color = vec4<f32>(result, 1.);
	let clip_space_pos = camera.view_proj * vec4<f32>(pos, 1.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
	return out;
}
";

const SPHERE_DATA_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

//struct TransformUniform {
//    model: mat4x4<f32>,
//    normal: mat4x4<f32>,
//}

struct SettingsUniform {
    radius: f32,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

//@group(1) @binding(0)
//var<uniform> transform: TransformUniform;
@group(1) @binding(0)
var<uniform> settings: SettingsUniform;
struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct DataInput {
    @location(1) position: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
	@location(0) world_pos: vec3<f32>,
	@location(1) center: vec3<f32>,
	@location(2) color: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    data: DataInput,
) -> VertexOutput {
    //let model_matrix = transform.model;
    //let normal_matrix = transform.normal;

    //let world_vector_pos = (model_matrix * vec4<f32>(vector_i.orig_position, 1.)).xyz;
    //// Do we want to scale a vector field if we scale its attached mesh?
    //let world_vector_arrow_t = (model_matrix * vec4<f32>(vector_i.orig_position + vector_i.arrow, 1.)).xyz - world_vector_pos;
    //let arrow_ampl = length(world_vector_arrow_t);
    //let world_vector_arrow = normalize(world_vector_arrow_t);

    //// We define the output we want to send over to frag shader
    var out: VertexOutput;

    let camera_right = normalize(vec3<f32>(camera.view_proj.x.x, camera.view_proj.y.x, camera.view_proj.z.x));
    let camera_up = normalize(vec3<f32>(camera.view_proj.x.y, camera.view_proj.y.y, camera.view_proj.z.y));
    let world_position = data.position + (model.position.x * camera_right + model.position.y * camera_up) * settings.radius;
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.world_pos = world_position;
    out.center = data.position;
    out.color = data.color;
    return out;
}

// function from :
// https://iquilezles.org/articles/intersectors/
fn dot2(v: vec3<f32>) -> f32 { return dot(v, v); }


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

fn sphIntersect( ro: vec3<f32>, rd: vec3<f32>, ce: vec3<f32>, ra: f32 ) -> vec2<f32>
{
    let oc = ro - ce;
    let b = dot( oc, rd );
    let c = dot( oc, oc ) - ra*ra;
    var h = b*b - c;
    if( h<0.0 ) { return vec2<f32>(-1.0); } // no intersection
    h = sqrt( h );
    return vec2<f32>( -b-h, -b+h );
}

struct FragOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    let ro = camera.view_pos.xyz;
	let rd = normalize(in.world_pos - camera.view_pos.xyz);
    let ce = in.center;
    let r = settings.radius;
    //let pa = in.orig_position;
    //let pb1 = in.orig_position + 0.5 * in.arrow * 0.1;
    //let pb2 = in.orig_position + in.arrow * 0.1;

    var out: FragOutput;

    let t = sphIntersect( ro, rd, ce, r);
    if(t.x < 0.0) {
        discard;
    }
	let pos = ro + t.x * rd;
	let normal = normalize(pos - ce);

	let light_dir = normalize(light.position - pos);
	let view_dir = normalize(camera.view_pos.xyz - pos);
	let half_dir = normalize(view_dir + light_dir);
	let F0 = vec3<f32>(0.04, 0.04, 0.04);
	let D = DistributionGGX(normal, half_dir, 0.15);
	let F = fresnelSchlick(dot(half_dir, normal), F0);
	let G = GeometrySmith(normal, view_dir, light_dir, 0.01);
	let f_ct = D * F * G / (4. * dot(view_dir, normal) * dot(light_dir, normal));
	let kd = 1.0;
	let lambertian = in.color;
	let result = (kd * lambertian + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);
	out.color = vec4<f32>(result, 1.);
	let clip_space_pos = camera.view_proj * vec4<f32>(pos, 1.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
	return out;
}
";
