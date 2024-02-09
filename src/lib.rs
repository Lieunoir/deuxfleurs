use std::collections::HashMap;
use std::iter;
use std::ops::{Deref, DerefMut};
use wgpu::util::DeviceExt;
use winit::event_loop::EventLoopBuilder;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    window::{Window, WindowBuilder},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod camera;
pub mod model;
mod picker;
pub mod point_cloud;
pub mod curve;
pub mod resources;
mod screenshot;
mod texture;
pub mod types;
mod ui;
mod util;
use camera::{Camera, CameraController, CameraUniform};
use model::DrawModel;
use point_cloud::PointCloud;
use curve::Curve;
use types::*;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding: u32,
    color: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding2: u32,
}

pub trait ModelContainer {
    fn get_model(&mut self, name: &str) -> Option<&mut model::Model>;
}

impl ModelContainer for HashMap<String, model::Model> {
    fn get_model(&mut self, name: &str) -> Option<&mut model::Model> {
        self.get_mut(name)
    }
}

pub struct State<'a> {
    // Graphic context
    surface: wgpu::Surface<'a>,
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
    // Camera
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    // 3D Model
    models: HashMap<String, model::Model>,
    //Points
    clouds: HashMap<String, PointCloud>,
    //Curves
    curves: HashMap<String, Curve>,
    //Volume meshes
    // Lighting
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    camera_light_bind_group_layout: wgpu::BindGroupLayout,
    camera_light_bind_group: wgpu::BindGroup,
    // egui
    //ui: ui::UI,

    // Item picker
    picker: picker::Picker,
}

pub struct StateWrapper<'a> {
    state: State<'a>,
    ui: ui::UI,
    callback: Option<Box<dyn FnMut(&mut egui::Ui, &mut State)>>,
}

pub enum UserEvent {
    LoadMesh(Vec<[f32; 3]>, SurfaceIndices),
    Pick,
}

impl<'a> State<'a> {
    // Initialize the state
    pub async fn new<'b: 'a>(window: &'b Window) -> State<'a> {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
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
                    required_features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                // Some(&std::path::Path::new("trace")), // Trace path
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
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
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 10,
        };
        surface.configure(&device, &config);

        // Bind the camera to the shaders
        let camera = Camera::new(config.width as f32 / config.height as f32);
        let camera_controller = CameraController::new();

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

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
            ],
            label: Some("camera_light_bind_group"),
        });

        let models = HashMap::new();
        let clouds = HashMap::new();
        let curves = HashMap::new();
        // Create depth texture
        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        // Create texture for screenshots
        let screenshoter =
            screenshot::Screenshoter::new(&device, size.width, size.height, surface_format);

        // Clear color used for mouse input interaction
        //let ui = ui::UI::new(&device, target_format, event_loop);
        let picker = picker::Picker::new(&device, size.width, size.height);
        Self {
            surface,
            device,
            queue,
            config,
            size,
            depth_texture,
            screenshoter,
            screenshot: false,
            camera,
            camera_controller,
            camera_buffer,
            camera_uniform,
            models,
            clouds,
            curves,
            light_uniform,
            light_buffer,
            camera_light_bind_group_layout,
            camera_light_bind_group,
            //ui,
            picker,
        }
    }

    // Keeps state in sync with window size when changed
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            // Make sure to current window size to depth texture - required for calc
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.screenshoter = screenshot::Screenshoter::new(
                &self.device,
                new_size.width,
                new_size.height,
                self.config.format,
            );
            self.camera.resize(new_size.width, new_size.height);
            self.picker
                .resize(&self.device, new_size.width, new_size.height);
        }
    }

    // Handle input using WindowEvent
    fn input(&mut self, event: &WindowEvent) -> bool {
        // Send any input to camera controller
        self.camera_controller.process_events(event);
        self.picker.input(event);
        false
    }

    fn update(&mut self) {
        // Sync local app state with camera
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
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
    }

    // Primary render flow
    fn render(
        &mut self,
        event_loop_proxy: &winit::event_loop::EventLoopProxy<crate::UserEvent>,
        ui: &mut ui::UI,
    ) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let view_ref = if !self.screenshot {
            &view
        } else {
            self.screenshoter.get_view()
        };

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

        for model in self.models.values_mut() {
            model.refresh_data(
                &self.device,
                &mut self.queue,
                &self.camera_light_bind_group_layout,
                self.config.format,
            );
        }

        for cloud in self.clouds.values_mut() {
            cloud.refresh_data(
                &self.device,
                &mut self.queue,
                &self.camera_light_bind_group_layout,
                self.config.format,
            );
        }

        for curve in self.curves.values_mut() {
            curve.refresh_data(
                &self.device,
                &mut self.queue,
                &self.camera_light_bind_group_layout,
                self.config.format,
            );
        }

        self.picker.render(
            &self.device,
            &mut encoder,
            &self.depth_texture.view,
            &self.camera_light_bind_group,
            &self.models,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: view_ref,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // Create a depth stencil buffer using the depth texture
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

            render_pass.set_bind_group(0, &self.camera_light_bind_group, &[]);

            // Draw the models
            for model in self.models.values() {
                render_pass.draw_model(model);
            }
            for cloud in self.clouds.values() {
                cloud.render(&mut render_pass);
                //render_pass.draw_model(cloud);
            }
            for curve in self.curves.values() {
                curve.render(&mut render_pass);
            }

            // Draw the gui
            if !self.screenshot {
                ui.render(&mut render_pass, &clipped_primitives, &screen_descriptor);
            }
        }

        if self.screenshot {
            self.screenshoter.copy_texture_to_buffer(&mut encoder);
            let index = self.queue.submit(iter::once(encoder.finish()));
            self.screenshoter.create_png(&self.device, index);
            self.screenshot = false;
        } else {
            self.queue.submit(
                user_cmd_bufs
                    .into_iter()
                    .chain(iter::once(encoder.finish())),
            );
            output.present();
        }

        self.picker.post_render(event_loop_proxy);
        Ok(())
    }

    pub fn register_mesh<V: Vertices, I: Into<SurfaceIndices>>(
        &mut self,
        mesh_name: &'_ str,
        //vertices: &Vec<[f32; 3]>,
        vertices: V,
        indices: I,
    ) -> &mut model::Model {
        let model = model::Model::new(
            mesh_name,
            vertices.into(),
            indices.into(),
            &self.device,
            &self.camera_light_bind_group_layout,
            &self.picker.bind_group_layout,
            self.config.format,
        );
        self.models.insert(mesh_name.into(), model);
        self.picker.counters_dirty = true;
        self.models.get_mut(mesh_name).unwrap()
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
            //&self.picker.bind_group_layout,
            self.config.format,
        );
        self.clouds.insert(name.clone(), model);
        self.clouds.get_mut(&name).unwrap()
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
            //&self.picker.bind_group_layout,
            self.config.format,
        );
        self.curves.insert(name.clone(), model);
        self.curves.get_mut(&name).unwrap()
    }

    pub fn screenshot(&mut self) {
        self.screenshot = true;
    }

    /*
    pub async fn register_mesh_from_path(&mut self, mesh_name: &'_ str, mesh_path: &'_ str) {
        // Bind the texture to the renderer
        // This creates a general texture bind group
        let texture_bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: Some("texture_bind_group_layout"),
                });
        let mut obj_model = resources::load_model(
            mesh_path,
            &self.device,
            &self.queue,
            &texture_bind_group_layout,
        )
        .await
        .expect("Couldn't load model. Maybe path is wrong?");
        obj_model.mesh.build_vertex_buffer(&self.device);
        self.models.insert(mesh_name.into(), obj_model);
    }
    */

    pub fn get_model(&mut self, name: &str) -> Option<&mut model::Model> {
        self.models.get_mut(name)
    }

    pub fn get_model_ref(&self, name: &str) -> Option<&model::Model> {
        self.models.get(name)
    }

    pub fn get_picked(&self) -> &Option<(String, usize)> {
        &self.picker.picked_item
    }
}

impl<'a> StateWrapper<'a> {
    pub async fn new<'b: 'a>(
        event_loop: &EventLoop<UserEvent>,
        window: &'b Window,
    ) -> StateWrapper<'a> {
        let state = State::new(window).await;
        let ui = ui::UI::new(
            &state.device,
            state.config.format,
            event_loop,
            window.scale_factor(),
        );
        Self {
            state,
            ui,
            callback: None,
        }
    }

    pub fn run(mut self, event_loop: EventLoop<UserEvent>, window: &Window) {
        let event_loop_proxy = event_loop.create_proxy();
        //event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run(move |event, elwt| {
            match event {
                Event::UserEvent(UserEvent::LoadMesh(mesh_v, mesh_f)) => {
                    self.state.register_mesh("loaded mesh", mesh_v, mesh_f);
                }
                Event::UserEvent(UserEvent::Pick) => {
                    self.state.picker.pick(&self.state.models);
                }
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == window.id() => {
                    let processed = self.ui.process_event(window, event);
                    if !processed && !self.state.input(event) {
                        // Handle window events (like resizing, or key inputs)
                        // This is stuff from `winit` -- see their docs for more info
                        match event {
                            WindowEvent::CloseRequested
                            | WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        logical_key: Key::Named(NamedKey::Escape),
                                        state: ElementState::Pressed,
                                        ..
                                    },
                                //KeyboardInput {
                                //    state: ElementState::Pressed,
                                //    virtual_keycode: Some(VirtualKeyCode::Escape),
                                //    ..
                                //},
                                ..
                            } => elwt.exit(),
                            WindowEvent::Resized(physical_size) => {
                                self.state.resize(*physical_size);
                            }
                            //WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                            //    // new_inner_size is &&mut so w have to dereference it twice
                            //    //self.state.resize(**new_inner_size);
                            //}
                            WindowEvent::RedrawRequested => {
                                self.state.update();
                                self.ui.draw_models(
                                    window,
                                    &mut self.state.models,
                                    &mut self.state.clouds,
                                    &mut self.state.curves,
                                    self.state.camera.build_view(),
                                    self.state.camera.build_proj(),
                                );
                                self.ui.draw_callback(
                                    &event_loop_proxy,
                                    &mut self.state,
                                    &mut self.callback,
                                );
                                match self.state.render(&event_loop_proxy, &mut self.ui) {
                                    Ok(()) => self.ui.handle_platform_output(window),
                                    // Reconfigure the surface if it's lost or outdated
                                    Err(
                                        wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated,
                                    ) => self.state.resize(self.state.size),
                                    // The system is out of memory, we should probably quit
                                    Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),

                                    Err(wgpu::SurfaceError::Timeout) => {
                                        log::warn!("Surface timeout")
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Event::AboutToWait => {
                    // RedrawRequested will only trigger once, unless we manually
                    // request it.
                    window.request_redraw();
                }
                _ => {}
            }
        });
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.state.resize(new_size)
    }

    /*pub fn register_mesh(
        &mut self,
        mesh_name: &'_ str,
        vertices: &Vec<[f32; 3]>,
        indices: &Vec<Vec<u32>>,
    ) -> &mut model::Model {
        self.state.register_mesh(mesh_name, vertices, indices)
    }*/

    /*
    pub fn register_vector_field(
        &mut self,
        vectors: Vec<[f32; 3]>,
        vectors_offsets: Vec<[f32; 3]>,
    ) {
        self.state.register_vector_field(vectors, vectors_offsets);
    }
    */

    //pub fn screenshot(&mut self) {
    //    self.state.screenshot();
    //}

    //pub fn get_model(&mut self, name: &str) -> Option<&mut model::Model> {
    //    self.state.get_model(name)
    //}

    pub fn set_callback<T>(&mut self, callback: T)
    where
        T: FnMut(&mut egui::Ui, &mut State) + 'static,
    {
        let boxed: Box<dyn FnMut(&mut egui::Ui, &mut State) + 'static> = Box::new(callback);
        self.callback = Some(boxed);
    }

    pub fn get_picked(&self) -> &Option<(String, usize)> {
        self.state.get_picked()
    }
}

impl<'a> Deref for StateWrapper<'a> {
    type Target = State<'a>;
    fn deref(&self) -> &State<'a> {
        &self.state
    }
}

impl<'a> DerefMut for StateWrapper<'a> {
    fn deref_mut(&mut self) -> &mut State<'a> {
        &mut self.state
    }
}

pub async fn create_window(width: u32, height: u32) -> (EventLoop<UserEvent>, Window) {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event()
        .build()
        .unwrap();
    let window = WindowBuilder::new()
        .with_title("Deuxfleurs")
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.body()?;
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

    (event_loop, window)
}
