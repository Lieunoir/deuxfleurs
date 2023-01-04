use crate::texture;
use crate::util;
use wgpu::util::DeviceExt;

pub struct VectorField {
    pub vectors: Vec<[f32; 3]>,
    pub vectors_offsets: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 3]>,
    pub render_pipeline: wgpu::RenderPipeline,
    //pub compute_pipeline: wgpu::ComputePipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub vector_buffer: wgpu::Buffer,
    pub billboard_buffer: wgpu::Buffer,
    //pub io_bind_group: wgpu::BindGroup,
}

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorVertex {
    pub position: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorData {
    pub color: [f32; 3],
    pub orig_position: [f32; 3],
    pub vector: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct BillboardTransform {
    pub transform: [[f32; 3]; 3],
    //pub _padding: [f32; 7],
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
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

impl Vertex for BillboardTransform {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

fn create_compute_transforms_pipeline(
    device: &wgpu::Device,
    camera_light_bind_group_layout: &wgpu::BindGroupLayout,
    io_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Vector Compute Pipeline Layout"),
        bind_group_layouts: &[camera_light_bind_group_layout, io_bind_group_layout],
        push_constant_ranges: &[],
    });
    let shader = wgpu::ShaderModuleDescriptor {
        label: Some("billboard shader"),
        source: wgpu::ShaderSource::Wgsl(super::arrow_shader::BILLBOARD_SHADER.into()),
    };
    let shader = device.create_shader_module(shader);
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Vector Render Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "cp_main",
    })
}

impl VectorField {
    fn build_vertex_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        let positions = [
            [0.485, 0., 0.],  //0
            [0.515, 0., 0.],  //1
            [0.485, 0.7, 0.], //2
            [0.515, 0., 0.],  //1
            [0.515, 0.7, 0.], //3
            [0.485, 0.7, 0.], //2
            [0.45, 0.7, 0.],  //arrow
            [0.55, 0.7, 0.],
            [0.5, 1.0, 0.],
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
        vectors_offsets: &Vec<[f32; 3]>,
        colors: &Vec<[f32; 3]>,
    ) -> wgpu::Buffer {
        let mut gpu_vertices = Vec::with_capacity(vectors.len());
        for ((vector, offset), color) in vectors.iter().zip(vectors_offsets).zip(colors) {
            let vertex = VectorData {
                color: *color,
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

    fn build_billboard_transform_buffer(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
        let gpu_vertices = vec![[[0.; 4]; 4]; len];
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vector Billboard Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        })
    }

    pub fn new(
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        vectors: Vec<[f32; 3]>,
        vectors_offsets: Vec<[f32; 3]>,
    ) -> Self {
        assert!(vectors.len() == vectors_offsets.len());
        let colors = vec![[1., 0., 0.]; vectors.len()];

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vector Render Pipeline Layout"),
            bind_group_layouts: &[
                camera_light_bind_group_layout,
                //transform_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("arrow shader"),
            source: wgpu::ShaderSource::Wgsl(super::arrow_shader::ARROW_SHADER.into()),
        };
        let render_pipeline = util::create_render_pipeline(
            device,
            &pipeline_layout,
            color_format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[
                VectorVertex::desc(),
                VectorData::desc(),
                BillboardTransform::desc(),
            ],
            shader,
            Some("vector field render"),
        );

        /*
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Vector Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[VectorVertex::desc(), VectorData::desc(), BillboardTransform::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        alpha: wgpu::BlendComponent::REPLACE,
                        color: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                //cull_mode: Some(wgpu::Face::Back),
                cull_mode: None,
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
        });
        */
        let vertex_buffer = Self::build_vertex_buffer(device);
        let vector_buffer = Self::build_vector_buffer(device, &vectors, &vectors_offsets, &colors);
        let billboard_buffer = Self::build_billboard_transform_buffer(device, vectors.len());
        /*
        let io_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage{
                                read_only: true,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage{
                                read_only: false,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    ],
                    label: Some("io_bind_group_layout"),
            });
        let io_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &io_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vector_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: billboard_buffer.as_entire_binding(),
                },
            ],
            label: Some("io_bind_group"),
        });
        let compute_pipeline = create_compute_transforms_pipeline(device, &camera_light_bind_group_layout, &io_bind_group_layout);
        */
        Self {
            vectors,
            vectors_offsets,
            colors,
            render_pipeline,
            //compute_pipeline,
            vertex_buffer,
            vector_buffer,
            billboard_buffer,
            //io_bind_group,
        }
    }

    /*
    pub fn update<'a, 'b>(&'a self, compute_pass: &mut wgpu::ComputePass<'b>)
    where
        'a: 'b,
    {
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(1, &self.io_bind_group, &[]);
        compute_pass.dispatch_workgroups(256, 1, 1);
    }
    */

    pub fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.vector_buffer.slice(..));
        render_pass.set_vertex_buffer(2, self.billboard_buffer.slice(..));
        render_pass.draw(0..9, 0..(self.vectors.len() as u32));
    }
}

// normale du rectangle pointe vers la camera
