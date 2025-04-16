#![doc = include_str!("../README.md")]
use crate::updater::Render;
use indexmap::IndexMap;
use pollster::FutureExt;
use rand::Rng;
use std::iter;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey, SmolStr};
use winit::window::WindowAttributes;
use winit::{dpi::PhysicalSize, event::*, event_loop::EventLoop, window::Window};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod aabb;
mod attachment;
mod camera;
pub mod curve;
mod data;
mod deferred;
mod obj_load;
mod picker;
pub mod point_cloud;
pub mod resources;
mod screenshot;
mod settings;
mod shader;
pub mod surface;
mod texture;
/// General types for genericity in functions parameters.
pub mod types;
mod ui;
mod updater;
mod util;
use camera::{Camera, CameraController, CameraUniform};
use curve::Curve;
use point_cloud::PointCloud;
use surface::Surface;
use types::*;

pub use egui;
pub use settings::Settings;
pub use wgpu::Color;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 3],
    _padding: u32,
    color: [f32; 3],
    _padding2: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct JitterUniform {
    x: f32,
    y: f32,
    _padding: [u32; 2],
}

pub struct State {
    settings: Settings,

    window: Arc<Window>,
    // Graphic context
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    // Window size
    size: winit::dpi::PhysicalSize<u32>,
    // Textures
    depth_texture: texture::Texture,
    // Screenshots
    screenshoter: screenshot::Screenshoter,
    screenshot: bool,

    // Keyboard
    ctrl_pressed: bool,
    // Camera
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    // 3D Model
    surfaces: IndexMap<String, Surface>,
    //Points
    clouds: IndexMap<String, PointCloud>,
    //Curves
    curves: IndexMap<String, Curve>,
    //Volume meshes
    // Lighting
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    jitter_buffer: wgpu::Buffer,
    camera_light_bind_group_layout: wgpu::BindGroupLayout,
    camera_light_bind_group: wgpu::BindGroup,
    // egui
    //ui: ui::UI,
    //time: std::time::Instant,
    dirty: bool,
    egui_dirty: bool,

    // Item picker
    picker: picker::Picker,

    copy: deferred::TextureCopy,
    pbr_renderer: deferred::PBR,
    ground: deferred::Ground,
    taa_counter: u8,
    aabb: aabb::SBV,
}

pub struct StateBuilder<T: FnOnce(&mut State), U: FnMut(&mut egui::Ui, &mut State)> {
    state: Option<State>,
    ui: Option<ui::UI>,
    width: u32,
    height: u32,
    proxy: EventLoopProxy<UserEvent>,
    settings: Settings,
    init: Option<T>,
    callback: Option<U>,
}

pub(crate) enum UserEvent {
    LoadMesh(Vec<[f32; 3]>, SurfaceIndices),
    Pick,
}

impl State {
    // Initialize the state
    async fn new(window: Window, width: u32, height: u32, settings: Settings) -> Self {
        let size = window.inner_size();
        let window = Arc::new(window);
        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // Select a device to use
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    required_features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        {
                            let mut limits = wgpu::Limits::default();
                            //limits.max_buffer_size = 1073741824;
                            limits.max_buffer_size = 2147483647;
                            limits
                        }
                    },
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        // Config for surface
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            //present_mode: surface_caps.present_modes[0],
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        window.request_inner_size(PhysicalSize::new(width, height));

        // Bind the camera to the shaders
        let camera = Camera::new(config.width as f32 / config.height as f32);
        let camera_controller = CameraController::new();

        let mut camera_uniform = CameraUniform::new();
        let aabb = aabb::SBV::default();
        camera_uniform.update_view_proj(&camera, &aabb, 0.);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Lighting
        // Create light uniforms and setup buffer for them
        let light_uniform = LightUniform {
            position: [2.0, 2.0, 2.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
            _padding2: 0,
        };

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let jitter_uniform = JitterUniform {
            x: 0.,
            y: 0.,
            _padding: [0; 2],
        };

        let jitter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Jitter Buffer"),
            contents: bytemuck::cast_slice(&[jitter_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create a bind group for camera buffer
        let camera_light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX
                            | wgpu::ShaderStages::FRAGMENT
                            | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("camera_light_bind_group_layout"),
            });

        let camera_light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_light_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: jitter_buffer.as_entire_binding(),
                },
            ],
            label: Some("camera_light_bind_group"),
        });

        let clouds = IndexMap::new();
        let curves = IndexMap::new();
        let surfaces = IndexMap::new();
        // Create depth texture
        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        // Create texture for screenshots
        let screenshoter = screenshot::Screenshoter::new(
            &device,
            size.width.max(1),
            size.height.max(1),
            surface_format,
        );

        let picker = picker::Picker::new(&device, size.width.max(1), size.height.max(1));

        let copy = deferred::TextureCopy::new(
            &device,
            surface_format,
            size.width.max(1),
            size.height.max(1),
        );
        let pbr_renderer = deferred::PBR::new(
            &device,
            surface_format,
            &depth_texture.view,
            &camera_light_bind_group_layout,
            size.width.max(1),
            size.height.max(1),
        );
        let ground = deferred::Ground::new(
            &device,
            surface_format,
            &depth_texture.view,
            &camera_light_bind_group_layout,
            0.,
        );
        Self {
            settings,
            window,
            surface,
            device,
            queue,
            config,
            size,
            depth_texture,
            screenshoter,
            screenshot: false,
            ctrl_pressed: false,
            camera,
            camera_controller,
            camera_buffer,
            camera_uniform,
            surfaces,
            clouds,
            curves,
            light_uniform,
            light_buffer,
            jitter_buffer,
            camera_light_bind_group_layout,
            camera_light_bind_group,
            picker,
            //time: std::time::Instant::now(),
            dirty: true,
            egui_dirty: true,
            copy,
            pbr_renderer,
            ground,
            taa_counter: 0,
            aabb: aabb::SBV::default(),
        }
    }

    fn set_floor(&mut self) {
        use crate::updater::Positions;
        let mut min_y = 0.;
        for surface in self.surfaces.values() {
            if surface.show {
                for p in surface.geometry.get_positions() {
                    let p = cgmath::Matrix4::from(surface.updater.transform.get_transform())
                        * cgmath::Point3::from(*p).to_homogeneous();
                    let p = p / p[3];
                    if p[1] < min_y {
                        min_y = p[1];
                    }
                }
            }
        }
        for cloud in self.clouds.values() {
            if cloud.show {
                for p in cloud.geometry.get_positions() {
                    let p = cgmath::Matrix4::from(cloud.updater.transform.get_transform())
                        * cgmath::Point3::from(*p).to_homogeneous();
                    let p = p / p[3];
                    if p[1] < min_y {
                        min_y = p[1];
                    }
                }
            }
        }
        for curve in self.curves.values() {
            if curve.show {
                for p in curve.geometry.get_positions() {
                    let p = cgmath::Matrix4::from(curve.updater.transform.get_transform())
                        * cgmath::Point3::from(*p).to_homogeneous();
                    let p = p / p[3];
                    if p[1] < min_y {
                        min_y = p[1];
                    }
                }
            }
        }
        self.ground.set_level(&mut self.queue, min_y);
    }

    /// Fit camera and ground to match the visible elements
    pub fn resize_scene(&mut self) {
        let mut size = None;
        let mut n = 0;
        let mut center = cgmath::Point3::<f32>::new(0., 0., 0.);
        for surface in self.surfaces.values() {
            if surface.show {
                let sbv = surface
                    .sbv
                    .transform(&surface.updater.transform.get_transform());
                center += sbv.center.into();
                n += 1;
                if let Some(size) = &mut size {
                    if sbv.radius > *size {
                        *size = sbv.radius;
                    }
                } else {
                    size = Some(sbv.radius);
                }
            }
        }
        for cloud in self.clouds.values() {
            if cloud.show {
                let sbv = cloud
                    .sbv
                    .transform(&cloud.updater.transform.get_transform());
                center += sbv.center.into();
                n += 1;
                if let Some(size) = &mut size {
                    if sbv.radius > *size {
                        *size = sbv.radius;
                    }
                } else {
                    size = Some(sbv.radius);
                }
            }
        }
        for curve in self.curves.values() {
            if curve.show {
                let sbv = curve
                    .sbv
                    .transform(&curve.updater.transform.get_transform());
                center += sbv.center.into();
                n += 1;
                if let Some(size) = &mut size {
                    if sbv.radius > *size {
                        *size = sbv.radius;
                    }
                } else {
                    size = Some(sbv.radius);
                }
            }
        }
        if n > 0 {
            center = center / (n as f32);
        }
        self.dirty = true;
        self.camera
            .set_scene_size(size.unwrap_or_else(|| 1.), center);
        self.set_floor();
    }

    // Keeps state in sync with window size when changed
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            // Make sure to current window size to depth texture - required for calc
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.screenshoter.resize(
                &self.device,
                new_size.width,
                new_size.height,
                self.config.format,
            );
            self.camera.resize(new_size.width, new_size.height);
            self.picker
                .resize(&self.device, new_size.width, new_size.height);
            self.copy.resize(
                &self.device,
                self.config.format,
                new_size.width,
                new_size.height,
            );
            self.pbr_renderer.resize(
                &self.device,
                self.config.format,
                &self.depth_texture.view,
                new_size.width,
                new_size.height,
            );
        }
    }

    // Handle input using WindowEvent
    fn input(&mut self, event: &WindowEvent) -> bool {
        // Send any input to camera controller
        let changed = self.camera_controller.process_events(event);
        self.dirty |= changed;
        self.picker.input(event) || changed
    }

    fn update(&mut self) -> bool {
        // Sync local app state with camera
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.aabb, self.ground.level);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update the light
        // TODO other optional light behaviors
        self.light_uniform.position = self.camera.get_position();
        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );

        let mut changed = self.dirty;

        let mut sbv = None;

        for surface in self.surfaces.values_mut() {
            changed |= surface.refresh(
                &self.device,
                &mut self.queue,
                &self.camera_light_bind_group_layout,
                self.config.format,
                &mut sbv,
            );
        }

        for cloud in self.clouds.values_mut() {
            changed |= cloud.refresh(
                &self.device,
                &mut self.queue,
                &self.camera_light_bind_group_layout,
                self.config.format,
                &mut sbv,
            );
        }

        for curve in self.curves.values_mut() {
            changed |= curve.refresh(
                &self.device,
                &mut self.queue,
                &self.camera_light_bind_group_layout,
                self.config.format,
                &mut sbv,
            );
        }
        self.aabb = sbv.unwrap_or_else(aabb::SBV::default);

        self.dirty = false;
        changed
    }

    // Primary render flow
    fn render(
        &mut self,
        event_loop_proxy: &winit::event_loop::EventLoopProxy<crate::UserEvent>,
        ui: &mut ui::UI,
        scene_changed: bool,
    ) -> Result<bool, wgpu::SurfaceError> {
        //println!("{}", self.time.elapsed().as_millis());
        //println!("{}", 1000. / (self.time.elapsed().as_millis()) as f32);
        //self.time = std::time::Instant::now();

        let mut request_redraw = false;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let (user_cmd_bufs, clipped_primitives, screen_descriptor) = ui.render_deltas(
            &self.device,
            &self.queue,
            &mut encoder,
            self.size.width,
            self.size.height,
        );

        self.egui_dirty |= self.picker.render(
            &self.device,
            &mut encoder,
            &self.depth_texture.view,
            &self.camera_light_bind_group,
            &self.surfaces,
            &self.clouds,
            &self.curves,
        );

        let output = if !self.screenshot {
            Some(self.surface.get_current_texture()?)
        } else {
            None
        };

        {
            let view = output.as_ref().map(|o| {
                o.texture
                    .create_view(&wgpu::TextureViewDescriptor::default())
            });

            let mut render = self.settings.rerender;
            let mut render_copy = false;
            let jitter;
            if scene_changed || !self.settings.taa.is_some() {
                // We rerender the scene from scratch
                request_redraw = self.settings.taa.is_some() && !self.settings.rerender;
                render = true;
                self.taa_counter = 0;
                jitter = JitterUniform {
                    x: 0.,
                    y: 0.,
                    _padding: [0; 2],
                };
            } else {
                if let Some(taa_frames) = self.settings.taa {
                    if self.taa_counter < taa_frames.get() {
                        // The scene hasn't changed but we need more copies for taa
                        render = true;
                        render_copy = true;
                        request_redraw = true;
                    }
                }

                let mut rng = rand::thread_rng();
                //let ampli = 0.5 + 0.5 * self.taa_counter as f32 / self.taa_frames as f32;
                let ampli = 1.;
                jitter = JitterUniform {
                    x: ampli * 2. * (rng.gen::<f32>() - 0.5) / self.size.width as f32,
                    y: ampli * 2. * (rng.gen::<f32>() - 0.5) / self.size.height as f32,
                    _padding: [0; 2],
                };
            };
            self.queue
                .write_buffer(&self.jitter_buffer, 0, bytemuck::cast_slice(&[jitter]));

            if render || self.screenshot {
                let view_ref = if self.screenshot {
                    self.copy.get_view()
                } else if render_copy {
                    self.taa_counter += 1;
                    self.copy.get_view()
                } else {
                    &view.as_ref().unwrap()
                };

                //Fix for clearing bug
                #[cfg(target_arch = "wasm32")]
                {
                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Material Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: self.pbr_renderer.get_albedo_view(),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                }

                let mut material_render_pass =
                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Material Render Pass"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: self.pbr_renderer.get_albedo_view(),
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.0,
                                        g: 0.0,
                                        b: 0.0,
                                        a: 0.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: self.pbr_renderer.get_normals_view(),
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.0,
                                        g: 0.0,
                                        b: 0.0,
                                        a: 0.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_texture.view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });

                material_render_pass.set_bind_group(0, &self.camera_light_bind_group, &[]);

                //order matters!
                //cloud discard so no depth test
                //curve only change deph sometimes, could use conservative depth
                //surface fully uses depth buffer(except attachments)
                for cloud in self.clouds.values() {
                    cloud.render(&mut material_render_pass);
                }
                for curve in self.curves.values() {
                    curve.render(&mut material_render_pass);
                }
                for surface in self.surfaces.values() {
                    surface.render(&mut material_render_pass);
                }
                drop(material_render_pass);

                let color = if !self.screenshot && !render_copy {
                    self.settings.color
                } else {
                    wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }
                };

                let mut pbr_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("PBR Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: view_ref,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                pbr_render_pass.set_bind_group(0, &self.camera_light_bind_group, &[]);
                self.pbr_renderer.render(&mut pbr_render_pass);
                drop(pbr_render_pass);
                if self.settings.shadow {
                    let mut shadow_render_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Shadow Render Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: self.ground.get_texture_view(),
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.,
                                        g: 0.,
                                        b: 0.,
                                        a: 0.,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });
                    shadow_render_pass.set_bind_group(0, &self.camera_light_bind_group, &[]);
                    for surface in self.surfaces.values() {
                        surface.render_shadow(&mut shadow_render_pass);
                    }

                    drop(shadow_render_pass);
                    self.ground
                        .blur(&mut encoder, &self.camera_light_bind_group);
                    let mut ground_render_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Shadow Render Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: view_ref,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            // Create a depth stencil buffer using the depth texture
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &self.depth_texture.view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });
                    ground_render_pass.set_bind_group(0, &self.camera_light_bind_group, &[]);
                    self.ground.render(&mut ground_render_pass);
                    drop(ground_render_pass);
                }

                // Draw the gui
                if !self.screenshot && !render_copy {
                    let ui_render_pass = encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Ui Render Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: view_ref,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            // Create a depth stencil buffer using the depth texture
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        })
                        .forget_lifetime();
                    ui.render(ui_render_pass, &clipped_primitives, &screen_descriptor);
                }
            }

            //do blending with previous frame
            if render_copy {
                let factor = (self.taa_counter as f64 - 1.) / (self.taa_counter as f64);
                self.copy.blend(&mut encoder, factor, self.taa_counter == 1);
            }

            if self.screenshot || (!render || render_copy) {
                let (view_ref, color) = if self.screenshot {
                    (
                        self.screenshoter.get_view(),
                        wgpu::Color {
                            r: 1.,
                            g: 1.,
                            b: 1.,
                            a: 0.,
                        },
                    )
                } else {
                    (view.as_ref().unwrap(), self.settings.color)
                };
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Copy Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: view_ref,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                self.copy.copy(&mut render_pass);

                let render_pass = render_pass.forget_lifetime();
                if !self.screenshot {
                    ui.render(render_pass, &clipped_primitives, &screen_descriptor);
                }
            }
        }

        if self.screenshot {
            self.screenshoter.copy_texture_to_buffer(&mut encoder);
            let index = self.queue.submit(iter::once(encoder.finish()));
            self.screenshoter
                .create_png(&self.device, index, self.config.format);
            self.screenshot = false;
        } else {
            self.queue.submit(
                user_cmd_bufs
                    .into_iter()
                    .chain(iter::once(encoder.finish())),
            );
            output.unwrap().present();
        }

        self.picker.post_render(event_loop_proxy);
        if self.picker.pick_locked || !self.settings.lazy_draw {
            request_redraw = true;
        }
        Ok(request_redraw)
    }

    pub fn register_surface<V: Vertices, I: Into<SurfaceIndices>>(
        &mut self,
        name: String,
        //vertices: &Vec<[f32; 3]>,
        vertices: V,
        indices: I,
    ) -> &mut Surface {
        let vertices: Vec<[f32; 3]> = vertices.into();
        if self.get_surface(&name).is_some()
            && self.get_surface(&name).unwrap().geometry.vertices.len() == vertices.len()
        {
            let mut surface = self.surfaces.shift_remove(&name).unwrap();
            surface.change_vertices(vertices, indices.into(), &self.device);
            self.surfaces.insert(name.clone(), surface);
        } else {
            let surface = Surface::new(
                name.clone(),
                vertices,
                indices.into(),
                &self.device,
                &self.camera_light_bind_group_layout,
                &self.picker.bind_group_layout,
                self.config.format,
            );
            self.surfaces.insert(name.clone(), surface);
            self.resize_scene();
            self.picker.counters_dirty = true;
        }
        self.dirty = true;
        self.surfaces.get_mut(&name).unwrap()
    }

    pub fn get_surface_mut(&mut self, name: &str) -> Option<&mut Surface> {
        self.dirty = true;
        self.surfaces.get_mut(name)
    }

    pub fn get_surface(&self, name: &str) -> Option<&Surface> {
        self.surfaces.get(name)
    }

    pub fn register_point_cloud<V: Vertices>(
        &mut self,
        name: String,
        positions: V,
    ) -> &mut PointCloud {
        let model = PointCloud::new(
            name.clone(),
            positions.into(),
            &self.device,
            &self.camera_light_bind_group_layout,
            &self.picker.bind_group_layout,
            self.config.format,
        );
        self.clouds.insert(name.clone(), model);
        self.picker.counters_dirty = true;
        self.dirty = true;
        self.resize_scene();
        self.clouds.get_mut(&name).unwrap()
    }

    pub fn get_point_cloud_mut(&mut self, name: &str) -> Option<&mut PointCloud> {
        self.dirty = true;
        self.clouds.get_mut(name)
    }

    pub fn get_point_cloud(&self, name: &str) -> Option<&PointCloud> {
        self.clouds.get(name)
    }

    pub fn register_curve<V: Vertices>(
        &mut self,
        name: String,
        positions: V,
        connections: Vec<[u32; 2]>,
    ) -> &mut Curve {
        let model = Curve::new(
            name.clone(),
            positions.into(),
            connections,
            &self.device,
            &self.camera_light_bind_group_layout,
            &self.picker.bind_group_layout,
            self.config.format,
        );
        self.curves.insert(name.clone(), model);
        self.picker.counters_dirty = true;
        self.dirty = true;
        self.resize_scene();
        self.curves.get_mut(&name).unwrap()
    }

    pub fn get_curve_mut(&mut self, name: &str) -> Option<&mut Curve> {
        self.dirty = true;
        self.curves.get_mut(name)
    }

    pub fn get_curve(&self, name: &str) -> Option<&Curve> {
        self.curves.get(name)
    }

    /// Take a screenshot at the next frame
    pub fn screenshot(&mut self) {
        self.screenshot = true;
    }

    /// Get current selected object: first the name, then index `i` of the selected element
    ///
    /// For a surface mesh, if `i` < `nv` then the selected element si the vertex of index `i`.
    /// If `nv` <= i < `nv + nf`, it corresponds to the face of index `i - nv`.
    pub fn get_picked(&self) -> &Option<(String, usize)> {
        &self.picker.picked_item
    }

    /// Politely ask to render the next frame
    pub fn refresh(&mut self) {
        self.dirty = true;
    }
}

impl<T: FnOnce(&mut State), U: FnMut(&mut egui::Ui, &mut State)> StateBuilder<T, U> {
    pub fn run(width: u32, height: u32, settings: Settings, init: Option<T>, callback: Option<U>) {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                std::panic::set_hook(Box::new(console_error_panic_hook::hook));
                console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
            } else {
                env_logger::init();
            }
        }

        let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
        let proxy = event_loop.create_proxy();
        let mut app = Self {
            state: None,
            ui: None,
            width,
            height,
            proxy,
            settings,
            init,
            callback,
        };
        #[cfg(not(target_arch = "wasm32"))]
        event_loop.run_app(&mut app).unwrap();
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::EventLoopExtWebSys;
            event_loop.spawn_app(app);
        }
    }
}

impl<T: FnOnce(&mut State), U: FnMut(&mut egui::Ui, &mut State)> ApplicationHandler<UserEvent>
    for StateBuilder<T, U>
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = WindowAttributes::default()
            .with_title("Deuxfleurs")
            .with_inner_size(PhysicalSize::new(self.width, self.height));
        let window = event_loop.create_window(window_attributes).unwrap();

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| {
                    //let dst = doc.body()?;
                    let dst = doc.get_element_by_id("wasm-load")?;
                    let canvas = window.canvas().unwrap();
                    // disable right click
                    let empty_func = js_sys::Function::new_no_args("return false;");
                    canvas.set_oncontextmenu(Some(&empty_func));
                    dst.append_child(&canvas).ok()?;
                    log::warn!("{}", canvas.width());
                    Some(())
                })
                .expect("Couldn't append canvas to document body.");
        }

        self.state =
            Some(State::new(window, self.width, self.height, self.settings.clone()).block_on());
        self.ui = Some(ui::UI::new(
            &self.state.as_ref().unwrap().device,
            event_loop,
            self.state.as_ref().unwrap().config.format,
            self.state.as_ref().unwrap().window.scale_factor(),
        ));
        self.init
            .take()
            .map(|f| f(&mut self.state.as_mut().unwrap()));
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        if let Some(state) = self.state.as_mut() {
            match event {
                UserEvent::LoadMesh(mesh_v, mesh_f) => {
                    state.register_surface("loaded mesh".into(), mesh_v, mesh_f);
                }
                UserEvent::Pick => {
                    state
                        .picker
                        .pick(&state.surfaces, &state.clouds, &state.curves, &state.camera);
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        //TODO the `if let` stuff is hacky and messy
        let processed = if let (Some(ui), Some(state)) = (self.ui.as_mut(), self.state.as_mut()) {
            if window_id == state.window.id() {
                ui.process_event(&*state.window, &event)
            } else {
                return;
            }
        } else {
            return;
        };

        let input = if !processed.consumed {
            if let Some(state) = self.state.as_mut() {
                state.input(&event)
            } else {
                false
            }
        } else {
            false
        };

        if processed.repaint {
            if let Some(state) = self.state.as_mut() {
                match event {
                    WindowEvent::RedrawRequested => {
                        if state.egui_dirty {
                            state.window.request_redraw();
                            state.egui_dirty = false;
                        } else {
                            state.egui_dirty = true;
                        }
                    }
                    _ => state.window.request_redraw(),
                }
            }
        }
        if !processed.consumed && !input {
            // Handle window events (like resizing, or key inputs)
            // This is stuff from `winit` -- see their docs for more info
            if let (Some(state), Some(ui)) = (self.state.as_mut(), self.ui.as_mut()) {
                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::Resized(physical_size) => {
                        state.resize(physical_size);
                        state.dirty = true;
                        state.window.request_redraw();
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key,
                                //state: ElementState::Pressed,
                                state: key_state,
                                ..
                            },
                        ..
                    } => {
                        if logical_key == Key::Named(NamedKey::Control) {
                            if key_state == ElementState::Pressed {
                                state.ctrl_pressed = true;
                            } else if key_state == ElementState::Released {
                                state.ctrl_pressed = false;
                            }
                        }
                        if state.ctrl_pressed
                            && logical_key == Key::Character(SmolStr::new_inline("c"))
                            && key_state == ElementState::Pressed
                        {
                            if let Ok(cam) = state.camera.copy() {
                                use clipboard::ClipboardProvider;
                                let _ = clipboard::ClipboardContext::new().map(|mut ctx| {
                                    let _ = ctx.set_contents(cam);
                                });
                            }
                        } else if state.ctrl_pressed
                            && logical_key == Key::Character(SmolStr::new_inline("v"))
                            && key_state == ElementState::Pressed
                        {
                            use clipboard::ClipboardProvider;
                            let _ = clipboard::ClipboardContext::new().map(|mut ctx| {
                                if let Ok(cam) = ctx.get_contents() {
                                    state.camera.set(cam);
                                    state.dirty = true;
                                }
                            });
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        //draw ui
                        ui.draw_models(
                            &*state.window,
                            &mut state.surfaces,
                            &mut state.clouds,
                            &mut state.curves,
                            state.camera.build_view(),
                            state.camera.build_proj(),
                        );
                        ui.draw_callback(&self.proxy, state, &mut self.callback);
                        let scene_changed = state.update();
                        //actual rendering
                        match state.render(&self.proxy, ui, scene_changed) {
                                            Ok(request_redraw) => {
                                                if request_redraw {
                                                    state.window.request_redraw();
                                                }
                                                ui.handle_platform_output(&*state.window)}
                                            ,
                                            // Reconfigure the surface if it's lost or outdated
                                            Err(
                                                wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated,
                                            ) => {
                                                state.resize(state.size)
                                            },
                                            // The system is out of memory, we should probably quit
                                            Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),

                                            Err(wgpu::SurfaceError::Timeout) => {
                                                log::warn!("Surface timeout")
                                            }
                                    }
                    }
                    _ => {}
                }
            }
        } else {
            if let Some(state) = self.state.as_mut() {
                state.window.request_redraw();
            }
        }
    }
}
