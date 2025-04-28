use super::{PCSettings, PointCloudGeometry};
use super::{SphereVertex, Vertex};
use crate::data::{DataUniform, DataUniformBuilder, TransformSettings};
use crate::texture;
use crate::updater::{ElementPicker, Render};
use crate::util::create_picker_pipeline;
use wgpu::util::DeviceExt;

pub struct Picker {
    num_elements: u32,
    vertex_buffer: wgpu::Buffer,
    center_buffer: wgpu::Buffer,
    transform_bind_group: wgpu::BindGroup,
    transform_buffer: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    settings_uniform: DataUniform,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexData {
    center: [f32; 3],
    index: u32,
}

impl Vertex for VertexData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
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
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

impl ElementPicker for Picker {
    type Geometry = PointCloudGeometry;
    type Settings = PCSettings;
    fn new(
        geometry: &Self::Geometry,
        settings: &Self::Settings,
        transform: &TransformSettings,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let s2 = 1.;
        let positions = [
            [-s2, -s2, 0.],
            [s2, -s2, 0.],
            [-s2, s2, 0.],
            [s2, -s2, 0.],
            [s2, s2, 0.],
            [-s2, s2, 0.],
        ];
        let vertices = positions.map(|position| SphereVertex { position });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let mut gpu_vertices = Vec::with_capacity(geometry.positions.len());
        for (i, position) in geometry.positions.iter().enumerate() {
            let vertex = VertexData {
                center: *position,
                index: i as u32,
            };
            gpu_vertices.push(vertex);
        }
        let center_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cloud Sphere Center Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let transform_raw = transform.to_raw();
        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Picker Transform Buffer"),
            contents: bytemuck::cast_slice(&[transform_raw]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let settings_uniform = settings.build_uniform(device).unwrap();
        let transform_bind_group_layout =
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
                label: Some("transform_bind_group_layout"),
            });

        let transform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &transform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: transform_buffer.as_entire_binding(),
            }],
            label: Some("transform_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Point Cloud Picker Pipeline Layout"),
                bind_group_layouts: &[
                    camera_light_bind_group_layout,
                    counter_bind_group_layout,
                    &transform_bind_group_layout,
                    &settings_uniform.bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("Point cloud Picker Shader"),
            source: wgpu::ShaderSource::Wgsl(SPHERE_PICKER_SHADER.into()),
        };
        let render_pipeline = create_picker_pipeline(
            device,
            &render_pipeline_layout,
            texture::Texture::PICKER_FORMAT,
            Some(texture::Texture::DEPTH_FORMAT),
            &[SphereVertex::desc(), VertexData::desc()],
            shader,
            Some("Point Cloud picker"),
        );
        let num_elements = geometry.positions.len() as u32;
        Self {
            num_elements,
            vertex_buffer,
            center_buffer,
            transform_bind_group,
            transform_buffer,
            render_pipeline,
            settings_uniform,
        }
    }

    fn update_transform(&self, queue: &mut wgpu::Queue, transform: &TransformSettings) {
        let transform_raw = transform.to_raw();
        queue.write_buffer(
            &self.transform_buffer,
            0,
            bytemuck::cast_slice(&[transform_raw]),
        );
    }

    fn update_settings(&self, queue: &mut wgpu::Queue, settings: &Self::Settings) {
        settings.refresh_buffer(queue, &self.settings_uniform);
    }

    fn get_total_elements(&self) -> u32 {
        self.num_elements
    }
}

impl Render for Picker {
    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(2, &self.transform_bind_group, &[]);
        render_pass.set_bind_group(3, &self.settings_uniform.bind_group, &[]);
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.center_buffer.slice(..));
        render_pass.draw(0..6, 0..(self.num_elements));
    }
}

const SPHERE_PICKER_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

struct TransformUniform {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
}

struct CounterUniform {
    count: u32,
    _padding_1: u32,
    _padding_2: u32,
    _padding_3: u32,
}

struct SettingsUniform {
    radius: f32,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

@group(1) @binding(0)
var<uniform> counter: CounterUniform;

@group(2) @binding(0)
var<uniform> transform: TransformUniform;
@group(3) @binding(0)
var<uniform> settings: SettingsUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct DataInput {
    @location(1) position: vec3<f32>,
    @location(2) index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
	@location(0) world_pos: vec3<f32>,
	@location(1) center: vec3<f32>,
    @location(2) index: u32,
};

@vertex
fn vs_main(
    model: VertexInput,
    data: DataInput,
) -> VertexOutput {
    let model_matrix = transform.model;

    //// We define the output we want to send over to frag shader
    var out: VertexOutput;

    let camera_right = normalize(vec3<f32>(camera.view_proj[0].x, camera.view_proj[1].x, camera.view_proj[2].x));
    let camera_up = normalize(vec3<f32>(camera.view_proj[0].y, camera.view_proj[1].y, camera.view_proj[2].y));
    //let world_position = (model_matrix * vec4<f32>(data.position, 1.)).xyz + normalize((model_matrix * vec4<f32>(model.position.x * camera_right + model.position.y * camera_up, 1.)).xyz) * settings.radius * sqrt(2.);
    let center = (model_matrix * vec4<f32>(data.position, 1.)).xyz;
    //let world_position = center + (model_matrix * vec4<f32>((model.position.x * camera_right + model.position.y * camera_up) * settings.radius, 1.)).xyz;
    let world_position = (model_matrix * vec4<f32>(data.position + (model.position.x * camera_right + model.position.y * camera_up) * settings.radius, 1.)).xyz;
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.world_pos = world_position;
    out.center = center;
    out.index = data.index;
    return out;
}

// function from :
// https://iquilezles.org/articles/intersectors/
fn dot2(v: vec3<f32>) -> f32 { return dot(v, v); }


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
    let det = determinant(transform.normal);
    let r = settings.radius / pow(det, 1. / 3.);

    var out: FragOutput;

    let t = sphIntersect( ro, rd, ce, r);
    if(t.x < 0.0) {
        discard;
    }
	let pos = ro + t.x * rd;

	let clip_space_pos = camera.view_proj * vec4<f32>(pos, 1.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
    let res =   counter.count + in.index;
    // webgl dosen't support rendering to u32, so we have to resort to this
    let f1 = f32((res >> u32(24))) / 255.;
    let f2 = f32(((res << u32(8)) >> u32(24))) / 255.;
    let f3 = f32(((res << u32(16)) >> u32(24))) / 255.;
    let f4 = f32(((res << u32(24)) >> u32(24))) / 255.;
    out.color = vec4<f32>(f4, f3, f2, f1);
    //return bitcast<vec4<f32>>(res);
	return out;
}
";
