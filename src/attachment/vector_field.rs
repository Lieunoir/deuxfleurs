use crate::data::*;
use crate::texture;
use crate::util;
use crate::util::Vertex;
use wgpu::util::DeviceExt;

pub struct VectorField {
    pub vectors: Vec<[f32; 3]>,
    pub _offsets: Vec<[f32; 3]>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vector_buffer: wgpu::Buffer,
    pub settings: VectorFieldSettings,
    pub settings_changed: bool,
    settings_bind_group: wgpu::BindGroup,
    settings_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VectorFieldSettingsRaw {
    magnitude: f32,
    _padding: [u32; 3],
    color: ColorSettings,
}

pub struct VectorFieldSettings {
    pub show: bool,
    pub magnitude: f32,
    pub color: ColorSettings,
}

impl Default for VectorFieldSettings {
    fn default() -> VectorFieldSettings {
        Self {
            show: true,
            magnitude: 1.,
            color: ColorSettings::default(),
        }
    }
}

impl VectorFieldSettings {
    fn to_raw(&self) -> VectorFieldSettingsRaw {
        VectorFieldSettingsRaw {
            magnitude: self.magnitude,
            _padding: [0; 3],
            color: self.color,
        }
    }

    pub fn set_magnitude(&mut self, magnitude: f32) {
        self.magnitude = magnitude;
    }

    pub fn show(&mut self, show: bool) {
        self.show = show;
    }

    pub fn set_color(&mut self, color: [f32; 4]) {
        self.color.color = color;
    }
}

pub(crate) struct NewVectorField {
    pub(crate) name: String,
    vectors: Vec<[f32; 3]>,
    offsets: Vec<[f32; 3]>,
    pub(crate) settings: VectorFieldSettings,
}

impl NewVectorField {
    pub(crate) fn new(
        name: String,
        vectors: Vec<[f32; 3]>,
        offsets: Vec<[f32; 3]>,
    ) -> NewVectorField {
        let mut settings = VectorFieldSettings::default();
        settings.color = ColorSettings::new(&name);
        NewVectorField {
            name,
            vectors,
            offsets,
            settings,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorVertex {
    pub position: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorData {
    pub orig_position: [f32; 3],
    pub vector: [f32; 3],
}

impl Vertex for VectorVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

impl Vertex for VectorData {
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
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

impl VectorField {
    fn build_vertex_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        let positions = [
            [-0.1, 0., -0.1],
            [0.1, 0., -0.1],
            [-0.1, 0., 0.1],
            [0.1, 0., 0.1],
            [-0.1, 1., 0.1],
            [0.1, 1., 0.1],
            [-0.1, 1., -0.1],
            [0.1, 1., -0.1],
        ];
        let vertices = positions.map(|position| VectorVertex { position });
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }

    fn build_vector_buffer(
        device: &wgpu::Device,
        vectors: &Vec<[f32; 3]>,
        offsets: &Vec<[f32; 3]>,
    ) -> wgpu::Buffer {
        let mut gpu_vertices = Vec::with_capacity(vectors.len());
        for (vector, offset) in vectors.iter().zip(offsets) {
            let vertex = VectorData {
                orig_position: *offset,
                vector: *vector,
            };
            gpu_vertices.push(vertex);
        }
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector Data Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        })
    }

    pub fn new(
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        NewVectorField {
            name: _,
            vectors,
            offsets,
            settings,
        }: NewVectorField,
    ) -> Self {
        assert!(vectors.len() == offsets.len());

        let settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector field settings buffer"),
            contents: bytemuck::cast_slice(&[settings.to_raw()]),
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
                label: Some("vector_field_settings_bind_group_layout"),
            });
        let settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &settings_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: settings_buffer.as_entire_binding(),
            }],
            label: Some("vector_field_settings_bind_group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vector Render Pipeline Layout"),
            bind_group_layouts: &[
                camera_light_bind_group_layout,
                transform_bind_group_layout,
                &settings_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("arrow shader"),
            source: wgpu::ShaderSource::Wgsl(super::vector_shader::ARROW_SHADER.into()),
        };
        let render_pipeline = util::create_quad_pipeline(
            device,
            &pipeline_layout,
            color_format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[VectorVertex::desc(), VectorData::desc()],
            shader,
            Some("vector field render"),
        );

        let vertex_buffer = Self::build_vertex_buffer(device);
        let vector_buffer = Self::build_vector_buffer(device, &vectors, &offsets);
        Self {
            vectors,
            _offsets: offsets,
            render_pipeline,
            vertex_buffer,
            vector_buffer,
            settings,
            settings_changed: false,
            settings_bind_group,
            settings_buffer,
        }
    }

    pub fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        if self.settings.show {
            render_pass.set_bind_group(2, &self.settings_bind_group, &[]);
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.vector_buffer.slice(..));
            //render_pass.draw(0..18, 0..(self.vectors.len() as u32));
            render_pass.draw(0..8, 0..(self.vectors.len() as u32));
        }
    }

    pub fn update(&mut self, queue: &mut wgpu::Queue) -> bool {
        if self.settings_changed {
            queue.write_buffer(
                &self.settings_buffer,
                0,
                bytemuck::cast_slice(&[self.settings.to_raw()]),
            );
            self.settings_changed = false;
            true
        } else {
            false
        }
    }
}
