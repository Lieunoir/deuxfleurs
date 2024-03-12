use crate::curve::Curve;
use crate::point_cloud::PointCloud;
use crate::surface::Surface;
use crate::texture;
use crate::util;
use crate::UserEvent;
use indexmap::IndexMap;
use wgpu::util::DeviceExt;
use winit::event::*;

pub struct Picker {
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    buffer: wgpu::Buffer,
    buffer_dimensions: util::BufferDimensions,
    pub picked_item: Option<(String, usize)>,
    item_to_pick: Option<(usize, usize)>,
    //lock to ensure buffer isn't used while mapped
    pick_locked: bool,
    dragging: bool,
    cur_pos: (f32, f32),
    orig_pos: (f32, f32),
    pub bind_group_layout: wgpu::BindGroupLayout,
    bind_groups: Vec<wgpu::BindGroup>,
    pub counters_dirty: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CounterUniform {
    count: u32,
    _padding_1: u32,
    _padding_2: u32,
    _padding_3: u32,
}

impl Picker {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let desc = wgpu::TextureDescriptor {
            label: Some("picker texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture::Texture::PICKER_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let buffer_dimensions = util::BufferDimensions::new::<u32>(width as usize, height as usize);
        let output_buffer_size = (buffer_dimensions.padded_bytes_per_row * buffer_dimensions.height)
            as wgpu::BufferAddress;
        let buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST
                // this tells wpgu that we want to read this buffer from the cpu
                | wgpu::BufferUsages::MAP_READ,
            label: Some("picker output buffer"),
            mapped_at_creation: false,
        };
        let buffer = device.create_buffer(&buffer_desc);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("picker_counter_bind_group_layout"),
        });
        /*
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: counter_buffer.as_entire_binding(),
                },
            ],
            label: Some("camera_light_bind_group"),
        });
        */
        let bind_groups = Vec::new();

        Self {
            texture,
            texture_view,
            buffer,
            buffer_dimensions,
            picked_item: None,
            item_to_pick: None,
            pick_locked: false,
            dragging: false,
            cur_pos: (0., 0.),
            orig_pos: (0., 0.),
            bind_group_layout,
            bind_groups,
            counters_dirty: true,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let desc = wgpu::TextureDescriptor {
            label: Some("picker texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::texture::Texture::PICKER_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        self.texture = device.create_texture(&desc);
        self.texture_view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let buffer_dimensions = util::BufferDimensions::new::<u32>(width as usize, height as usize);
        let output_buffer_size = (buffer_dimensions.padded_bytes_per_row * buffer_dimensions.height)
            as wgpu::BufferAddress;
        self.buffer_dimensions = buffer_dimensions;
        let picker_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST
                // this tells wpgu that we want to read this buffer from the cpu
                | wgpu::BufferUsages::MAP_READ,
            label: Some("picker output buffer"),
            mapped_at_creation: false,
        };
        self.buffer = device.create_buffer(&picker_buffer_desc);
        self.item_to_pick = None;
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cur_pos = (position.x as f32, position.y as f32);
                let dx = self.cur_pos.0 - self.orig_pos.0;
                let dy = self.cur_pos.1 - self.orig_pos.1;
                if dx * dx + dy * dy > 5. {
                    self.dragging = true;
                }
                false
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    if *state == ElementState::Pressed {
                        self.dragging = false;
                        self.orig_pos = self.cur_pos;
                    } else if !self.dragging {
                        self.item_to_pick =
                            Some((self.cur_pos.0 as usize, self.cur_pos.1 as usize));
                    }
                }
                true
            }
            _ => false,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth_texture_view: &wgpu::TextureView,
        camera_light_bind_group: &wgpu::BindGroup,
        surfaces: &IndexMap<String, Surface>,
        clouds: &IndexMap<String, PointCloud>,
        curves: &IndexMap<String, Curve>,
    ) -> bool {
        if !self.pick_locked && self.item_to_pick.is_some() {
            {
                let tex_view = &self.texture_view;
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Picker Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: tex_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Set the clear color during redraw
                            // This is basically a background color applied if an object isn't taking up space

                            // A standard clear color
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    // Create a depth stencil buffer using the depth texture
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_texture_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                render_pass.set_bind_group(0, camera_light_bind_group, &[]);

                if self.counters_dirty {
                    let mut counter = 1;
                    self.bind_groups = surfaces
                        .values()
                        .map(|surface| {
                            let counter_uniform = CounterUniform {
                                count: counter,
                                _padding_1: 0,
                                _padding_2: 0,
                                _padding_3: 0,
                            };
                            counter += surface.get_total_elements();

                            //TODO use one dynamic buffer instead
                            let counter_buffer =
                                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("counter buffer"),
                                    contents: bytemuck::cast_slice(&[counter_uniform]),
                                    usage: wgpu::BufferUsages::UNIFORM
                                        | wgpu::BufferUsages::COPY_DST,
                                });
                            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                layout: &self.bind_group_layout,
                                entries: &[wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: counter_buffer.as_entire_binding(),
                                }],
                                label: Some("camera_light_bind_group"),
                            });
                            bind_group
                        })
                        .collect();

                    for cloud in clouds.values() {
                        let counter_uniform = CounterUniform {
                            count: counter,
                            _padding_1: 0,
                            _padding_2: 0,
                            _padding_3: 0,
                        };
                        counter += cloud.get_total_elements();

                        let counter_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("counter buffer"),
                                contents: bytemuck::cast_slice(&[counter_uniform]),
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            });
                        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &self.bind_group_layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: counter_buffer.as_entire_binding(),
                            }],
                            label: Some("camera_light_bind_group"),
                        });
                        self.bind_groups.push(bind_group);
                    }

                    for curve in curves.values() {
                        let counter_uniform = CounterUniform {
                            count: counter,
                            _padding_1: 0,
                            _padding_2: 0,
                            _padding_3: 0,
                        };
                        counter += curve.get_total_elements();

                        let counter_buffer =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("counter buffer"),
                                contents: bytemuck::cast_slice(&[counter_uniform]),
                                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            });
                        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            layout: &self.bind_group_layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: counter_buffer.as_entire_binding(),
                            }],
                            label: Some("camera_light_bind_group"),
                        });
                        self.bind_groups.push(bind_group);
                    }

                    self.counters_dirty = false;
                }

                let mut index = 0;
                for surface in surfaces.values() {
                    let counter_bind_group = &self.bind_groups[index];
                    index += 1;
                    render_pass.set_bind_group(1, counter_bind_group, &[]);
                    surface.render_picker(&mut render_pass);
                }
                for cloud in clouds.values() {
                    let counter_bind_group = &self.bind_groups[index];
                    index += 1;
                    render_pass.set_bind_group(1, counter_bind_group, &[]);
                    cloud.render_picker(&mut render_pass);
                }
                for curve in curves.values() {
                    let counter_bind_group = &self.bind_groups[index];
                    index += 1;
                    render_pass.set_bind_group(1, counter_bind_group, &[]);
                    curve.render_picker(&mut render_pass);
                }
            }
            {
                encoder.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture {
                        aspect: wgpu::TextureAspect::All,
                        texture: &self.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                    },
                    wgpu::ImageCopyBuffer {
                        buffer: &self.buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(self.buffer_dimensions.padded_bytes_per_row as u32),
                            //rows_per_image: std::num::NonZeroU32::new(self.size.height),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d {
                        width: self.buffer_dimensions.width as u32,
                        height: self.buffer_dimensions.height as u32,
                        depth_or_array_layers: 1,
                    },
                );
            }
            true
        } else {
            false
        }
    }

    pub fn post_render(&mut self, event_loop_proxy: &winit::event_loop::EventLoopProxy<UserEvent>) {
        if !self.pick_locked && self.item_to_pick.is_some() {
            let buffer_slice = self.buffer.slice(..);
            let event_loop_proxy = event_loop_proxy.clone();

            #[cfg(not(target_arch = "wasm32"))]
            {
                buffer_slice.map_async(wgpu::MapMode::Read, move |_v| {
                    event_loop_proxy.send_event(UserEvent::Pick).ok();
                });
            }
            #[cfg(target_arch = "wasm32")]
            {
                // `event_loop_proxy` isn't `Send` so we have to find another way
                let (sender, receiver) = oneshot::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |_v| {
                    sender.send(()).unwrap();
                });
                wasm_bindgen_futures::spawn_local(async move {
                    if let Ok(()) = receiver.await {
                        event_loop_proxy.send_event(UserEvent::Pick).ok();
                    }
                });
            }
            //buffer_slice = Some(buffer_slice_t);
            self.pick_locked = true;
        }
    }

    pub fn pick(
        &mut self,
        surfaces: &IndexMap<String, Surface>,
        clouds: &IndexMap<String, PointCloud>,
        curves: &IndexMap<String, Curve>,
    ) {
        {
            let buffer_slice = self.buffer.slice(..);
            let data = buffer_slice.get_mapped_range();
            if let Some((i, j)) = self.item_to_pick {
                let index = j * self.buffer_dimensions.padded_bytes_per_row + 4 * i;
                let value = (data[index + 3] as u32) << 24
                    | (data[index + 2] as u32) << 16
                    | (data[index + 1] as u32) << 8
                    | (data[index] as u32);
                let mut c = 1;
                if let Some(name) = surfaces
                    .iter()
                    .find(|(_key, surface)| {
                        let found = c <= value && value < c + surface.get_total_elements();
                        if !found {
                            c += surface.get_total_elements();
                        }
                        found
                    })
                    .map(|(n, _s)| n)
                    .or_else(|| {
                        clouds
                            .iter()
                            .find(|(_key, cloud)| {
                                let found = c <= value && value < c + cloud.get_total_elements();
                                if !found {
                                    c += cloud.get_total_elements();
                                }
                                found
                            })
                            .map(|(n, _pc)| n)
                    })
                    .or_else(|| {
                        curves
                            .iter()
                            .find(|(_key, curve)| {
                                let found = c <= value && value < c + curve.get_total_elements();
                                if !found {
                                    c += curve.get_total_elements();
                                }
                                found
                            })
                            .map(|(n, _pc)| n)
                    })
                {
                    self.picked_item = Some((name.clone(), (value - c) as usize));
                } else {
                    self.picked_item = None;
                }
                self.item_to_pick = None;
            }
        }
        self.buffer.unmap();
        self.pick_locked = false;
    }
}
