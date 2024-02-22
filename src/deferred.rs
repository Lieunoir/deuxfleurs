use wgpu::util::DeviceExt;
use crate::util;
use crate::texture;

pub struct TextureCopy {
    square: wgpu::Buffer,
    copy_bind_group: wgpu::BindGroup,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    copy_pipeline: wgpu::RenderPipeline,
    copy_texture: wgpu::Texture,
    copy_texture_view: wgpu::TextureView,
    pub dirty: bool,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SquareVertex {
    position: [f32; 3],
}

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

impl Vertex for SquareVertex {
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

impl TextureCopy {
    //resize
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32) {
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        self.copy_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1, // We'll talk about this a little later
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("copy_texture"),
                view_formats: &[],
            }
        );
        self.copy_texture_view = self.copy_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let copy_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        self.copy_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &self.copy_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.copy_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&copy_sampler),
                    }
                ],
                label: Some("copy_bind_group"),
            }
        );
    }

    pub fn get_view(&self) -> &wgpu::TextureView {
        &self.copy_texture_view
    }

    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32
    ) -> Self {
        let positions = [
            [-1., -1., 0.],
            [1., -1., 0.],
            [-1., 1., 0.],
            [1., 1., 0.],
        ];
        let vertices = positions.map(|position| SquareVertex { position });
        let square = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Copy Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let copy_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1, // We'll talk about this a little later
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("copy_texture"),
                view_formats: &[],
            }
        );
        let copy_texture_view = copy_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let copy_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let copy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let copy_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &copy_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&copy_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&copy_sampler),
                    }
                ],
                label: Some("copy_bind_group"),
            }
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Copy Pipeline Layout"),
            bind_group_layouts: &[&copy_bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("copy shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        };
        let copy_pipeline = util::create_quad_pipeline(
            device,
            &pipeline_layout,
            color_format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[SquareVertex::desc()],
            shader,
            Some("copy render"),
        );


        Self {
            square,
            copy_bind_group,
            copy_bind_group_layout,
            copy_pipeline,
            copy_texture,
            copy_texture_view,
            dirty: true,
        }
    }

    pub fn copy<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(0, &self.copy_bind_group, &[]);
        render_pass.set_pipeline(&self.copy_pipeline);
        render_pass.set_vertex_buffer(0, self.square.slice(..));
        render_pass.draw(0..4, 0..1);
    }
}

const SHADER: &str = "
@group(0) @binding(0)
var t_copy: texture_2d<f32>;
@group(0) @binding(1)
var s_copy: sampler;


struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    ) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = vec2<f32>(model.position.x * 0.5 + 0.5, 1. - model.position.y * 0.5 - 0.5);
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_copy, s_copy, in.tex_coords);
}
";
