#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]
use crate::aabb::SBV;
use crate::private::InnerGraphicalState;
use crate::updater::Render;
#[cfg(not(target_arch = "wasm32"))]
use egui_winit::clipboard::Clipboard;
use indexmap::IndexMap;
use pollster::FutureExt;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::iter;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use web_sys::Clipboard;
use wgpu::rwh::HasDisplayHandle;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey, SmolStr};
use winit::window::WindowAttributes;
use winit::{dpi::PhysicalSize, event::*, event_loop::EventLoop, window::Window};
mod aabb;
pub mod attachment;
mod camera;
pub mod data;
mod deferred;
mod obj_load;
mod picker;
mod point_cloud;
mod resources;
mod screenshot;
mod segment;
mod settings;
mod shader;
mod surface;
mod texture;
/// General types for genericity in functions parameters.
pub mod types;
///  Custom Ui components for mesh loading
pub mod ui;
mod updater;
mod util;
use crate::point_cloud::DisplayPointCloud;
use crate::segment::DisplaySegment;
use crate::surface::DisplaySurface;
use camera::{Camera, CameraController, CameraUniform};
pub use egui;
pub use resources::{load_mesh, load_mesh_blocking};
pub use settings::Settings;
use types::*;
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

/// Generic structs re exported here for visibility. Refer to aliases for documentation
pub mod internal {
    //pub use crate::private::InnerState;
    pub use crate::updater::{Element, ElementMut};
}

mod private {
    use crate::aabb;
    use crate::attachment::{NewVectorField, VectorField};
    use crate::data::{DataSettings, DataUniformBuilder};
    use crate::deferred;
    use crate::point_cloud::{
        DisplayPointCloud, PointCloudDataBuffer, PointCloudFixedRenderer, PointCloudPipeline,
        UninitedPointCloud,
    };
    use crate::screenshot;
    use crate::segment::{
        DisplaySegment, SegmentDataBuffer, SegmentFixedRenderer, SegmentPipeline, UninitedSegment,
    };
    use crate::surface::{
        DisplaySurface, SurfaceDataBuffer, SurfaceFixedRenderer, SurfacePipeline, UninitedSurface,
    };
    use crate::texture;
    use crate::types::SurfaceIndices;
    use crate::ui::UiDataElement;
    use crate::updater::{
        AttachedGeometry, DataBuffer, DisplayElement, Element, ElementGeometry, ElementMut,
        FixedRenderer, GraphicalContext, NamedSettings, NewAttachedGeometry, RenderPipeline,
        Renderer, UninitedElement,
    };
    use crate::Camera;
    use crate::CameraController;
    use crate::CameraUniform;
    use crate::LightUniform;
    use crate::Settings;
    use crate::UserEvent;
    use crate::{picker, PointCloud, Segment, Surface};
    use indexmap::IndexMap;
    use rand::rngs::SmallRng;
    use std::sync::Arc;
    use winit::event_loop::EventLoopProxy;
    use winit::window::Window;

    pub trait ContextHolder {
        type Context<'a>;
        type SurfaceRenderer;
        type SurfaceAttachedData;
        type PointCloudRenderer;
        type PointCloudAttachedData;
        type SegmentRenderer;
        type SegmentAttachedData;
    }

    pub trait ContainerContextGiver<Element>: ContextHolder {
        fn get_container_mut(&mut self) -> (&mut IndexMap<String, Element>, Self::Context<'_>);
        fn get_container(&self) -> &IndexMap<String, Element>;
    }

    pub trait GeometryHolder<Element>: ContainerContextGiver<Element> {
        type Args;

        fn register(
            &mut self,
            name: String,
            args: Self::Args,
        ) -> ElementMut<'_, Element, Self::Context<'_>>;

        fn get_element_mut(
            &mut self,
            name: &str,
        ) -> Option<ElementMut<'_, Element, Self::Context<'_>>>;

        fn get_element(&self, name: &str) -> Option<&'_ Element>;
    }

    impl<Geometry, Settings, Data, Attached, T>
        GeometryHolder<UninitedElement<Geometry, Settings, Data, Attached>> for T
    where
        for<'a> T: ContextHolder<Context<'a> = ()>,
        T: ContainerContextGiver<UninitedElement<Geometry, Settings, Data, Attached>>,
        Geometry: ElementGeometry,
        Settings: DataUniformBuilder + NamedSettings,
        Attached: AttachedGeometry + NewAttachedGeometry,
    {
        type Args = Geometry::Args;

        fn register(
            &mut self,
            name: String,
            args: Self::Args,
        ) -> ElementMut<UninitedElement<Geometry, Settings, Data, Attached>, Self::Context<'_>>
        {
            let element = Element::new_bare(name.clone(), args);
            let container = self.get_container_mut().0;
            container.insert(name.clone(), element);
            ElementMut {
                element: container.get_mut(&name).unwrap(),
                context: (),
            }
        }

        fn get_element_mut(
            &mut self,
            name: &str,
        ) -> Option<
            ElementMut<UninitedElement<Geometry, Settings, Data, Attached>, Self::Context<'_>>,
        > {
            self.get_container_mut()
                .0
                .get_mut(name)
                .map(|element| ElementMut {
                    element,
                    context: (),
                })
        }

        fn get_element(
            &self,
            name: &str,
        ) -> Option<&'_ UninitedElement<Geometry, Settings, Data, Attached>> {
            self.get_container().get(name)
        }
    }

    impl<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached, T>
        GeometryHolder<DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>>
        for T
    where
        for<'a> T: ContextHolder<Context<'a> = GraphicalContext<'a>>,
        T: ContainerContextGiver<
            DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>,
        >,
        Attached: AttachedGeometry,
        Geometry: ElementGeometry,
        Data: DataUniformBuilder + DataSettings + UiDataElement,
        Settings: NamedSettings,
        Fixed: FixedRenderer<Settings = Settings, Data = Data, Geometry = Geometry>,
        DataB: DataBuffer<Settings = Settings, Data = Data, Geometry = Geometry>,
        Pipeline:
            RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry, Fixed = Fixed>,
    {
        type Args = Geometry::Args;

        fn register(
            &mut self,
            name: String,
            args: Self::Args,
        ) -> ElementMut<
            DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>,
            Self::Context<'_>,
        > {
            let (container, context) = self.get_container_mut();
            let element = Element::new(
                name.clone(),
                args,
                context.device,
                context.camera_light_bind_group_layout,
                context.counter_bind_group_layout,
                context.color_format,
            );
            container.insert(name.clone(), element);
            ElementMut {
                element: container.get_mut(&name).unwrap(),
                context,
            }
        }

        fn get_element_mut(
            &mut self,
            name: &str,
        ) -> Option<
            ElementMut<
                DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>,
                Self::Context<'_>,
            >,
        > {
            let (container, context) = self.get_container_mut();
            container
                .get_mut(name)
                .map(|element| ElementMut { element, context })
        }

        fn get_element(
            &self,
            name: &str,
        ) -> Option<&DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>>
        {
            self.get_container().get(name)
        }
    }

    /// Holds the application state. Starting point to add visualization datas.
    pub struct InnerGraphicalState {
        pub(crate) surfaces: IndexMap<String, DisplaySurface>,
        pub(crate) clouds: IndexMap<String, DisplayPointCloud>,
        pub(crate) segments: IndexMap<String, DisplaySegment>,
        pub(crate) settings: Settings,

        pub(crate) window: Arc<Window>,
        pub(crate) proxy: EventLoopProxy<UserEvent>,
        // Graphic context
        pub(crate) surface: wgpu::Surface<'static>,
        pub(crate) device: wgpu::Device,
        pub(crate) queue: wgpu::Queue,
        pub(crate) config: wgpu::SurfaceConfiguration,
        // Window size
        pub(crate) size: winit::dpi::PhysicalSize<u32>,
        // Textures
        pub(crate) depth_texture: texture::Texture,
        // Screenshots
        pub(crate) screenshoter: screenshot::Screenshoter,
        pub(crate) screenshot: bool,

        // Keyboard
        pub(crate) ctrl_pressed: bool,
        // Camera
        pub(crate) camera: Camera,
        pub(crate) camera_controller: CameraController,
        pub(crate) camera_uniform: CameraUniform,
        pub(crate) camera_buffer: wgpu::Buffer,
        // Lighting
        pub(crate) light_uniform: LightUniform,
        pub(crate) light_buffer: wgpu::Buffer,
        pub(crate) jitter_buffer: wgpu::Buffer,
        pub(crate) camera_light_bind_group_layout: wgpu::BindGroupLayout,
        pub(crate) camera_light_bind_group: wgpu::BindGroup,
        // egui
        //ui: ui::UI,
        //time: std::time::Instant,
        pub(crate) dirty: bool,
        pub(crate) egui_dirty: bool,
        pub(crate) should_resize: bool,

        // Item picker
        pub(crate) picker: picker::Picker,

        pub(crate) copy: deferred::TextureCopy,
        pub(crate) pbr_renderer: deferred::PBR,
        pub(crate) ground: deferred::Ground,
        pub(crate) taa_counter: u8,
        pub(crate) aabb: aabb::SBV,
        pub(crate) rng: SmallRng,
    }

    impl ContextHolder for InnerGraphicalState {
        type Context<'a> = GraphicalContext<'a>;
        type SurfaceRenderer = Renderer<SurfaceFixedRenderer, SurfaceDataBuffer, SurfacePipeline>;
        type SurfaceAttachedData = VectorField;
        type PointCloudRenderer =
            Renderer<PointCloudFixedRenderer, PointCloudDataBuffer, PointCloudPipeline>;
        type PointCloudAttachedData = ();
        type SegmentRenderer = Renderer<SegmentFixedRenderer, SegmentDataBuffer, SegmentPipeline>;
        type SegmentAttachedData = ();
    }

    impl ContainerContextGiver<DisplaySurface> for InnerGraphicalState {
        fn get_container(&self) -> &IndexMap<String, DisplaySurface> {
            &self.surfaces
        }

        fn get_container_mut(
            &mut self,
        ) -> (&mut IndexMap<String, DisplaySurface>, Self::Context<'_>) {
            (
                &mut self.surfaces,
                Self::Context {
                    settings: &self.settings,
                    device: &self.device,
                    queue: &self.queue,
                    camera_light_bind_group_layout: &self.camera_light_bind_group_layout,
                    counter_bind_group_layout: &self.picker.bind_group_layout,
                    color_format: self.config.format,
                    refresh_screen: &mut self.dirty,
                },
            )
        }
    }

    impl ContainerContextGiver<DisplayPointCloud> for InnerGraphicalState {
        fn get_container(&self) -> &IndexMap<String, DisplayPointCloud> {
            &self.clouds
        }

        fn get_container_mut(
            &mut self,
        ) -> (&mut IndexMap<String, DisplayPointCloud>, Self::Context<'_>) {
            (
                &mut self.clouds,
                Self::Context {
                    settings: &self.settings,
                    device: &self.device,
                    queue: &self.queue,
                    camera_light_bind_group_layout: &self.camera_light_bind_group_layout,
                    counter_bind_group_layout: &self.picker.bind_group_layout,
                    color_format: self.config.format,
                    refresh_screen: &mut self.dirty,
                },
            )
        }
    }

    impl ContainerContextGiver<DisplaySegment> for InnerGraphicalState {
        fn get_container(&self) -> &IndexMap<String, DisplaySegment> {
            &self.segments
        }

        fn get_container_mut(
            &mut self,
        ) -> (&mut IndexMap<String, DisplaySegment>, Self::Context<'_>) {
            (
                &mut self.segments,
                Self::Context {
                    settings: &self.settings,
                    device: &self.device,
                    queue: &self.queue,
                    camera_light_bind_group_layout: &self.camera_light_bind_group_layout,
                    counter_bind_group_layout: &self.picker.bind_group_layout,
                    color_format: self.config.format,
                    refresh_screen: &mut self.dirty,
                },
            )
        }
    }

    impl StateTrait for InnerGraphicalState {}

    pub struct InnerBareState {
        pub(crate) surfaces: IndexMap<String, UninitedSurface>,
        pub(crate) clouds: IndexMap<String, UninitedPointCloud>,
        pub(crate) segments: IndexMap<String, UninitedSegment>,
    }

    impl ContextHolder for InnerBareState {
        type Context<'a> = ();
        type SurfaceRenderer = ();
        type SurfaceAttachedData = NewVectorField;
        type PointCloudRenderer = ();
        type PointCloudAttachedData = ();
        type SegmentRenderer = ();
        type SegmentAttachedData = ();
    }

    impl ContainerContextGiver<UninitedSurface> for InnerBareState {
        fn get_container(&self) -> &IndexMap<String, UninitedSurface> {
            &self.surfaces
        }

        fn get_container_mut(&mut self) -> (&mut IndexMap<String, UninitedSurface>, ()) {
            (&mut self.surfaces, ())
        }
    }

    impl ContainerContextGiver<UninitedPointCloud> for InnerBareState {
        fn get_container(&self) -> &IndexMap<String, UninitedPointCloud> {
            &self.clouds
        }

        fn get_container_mut(&mut self) -> (&mut IndexMap<String, UninitedPointCloud>, ()) {
            (&mut self.clouds, ())
        }
    }

    impl ContainerContextGiver<UninitedSegment> for InnerBareState {
        fn get_container(&self) -> &IndexMap<String, UninitedSegment> {
            &self.segments
        }

        fn get_container_mut(&mut self) -> (&mut IndexMap<String, UninitedSegment>, ()) {
            (&mut self.segments, ())
        }
    }

    impl StateTrait for InnerBareState {}

    pub trait StateTrait:
        ContextHolder
        + GeometryHolder<
            Surface<
                <Self as ContextHolder>::SurfaceRenderer,
                <Self as ContextHolder>::SurfaceAttachedData,
            >,
            Args = (SurfaceIndices, Vec<[f32; 3]>),
        > + GeometryHolder<
            PointCloud<
                <Self as ContextHolder>::PointCloudRenderer,
                <Self as ContextHolder>::PointCloudAttachedData,
            >,
            Args = Vec<[f32; 3]>,
        > + GeometryHolder<
            Segment<
                <Self as ContextHolder>::SegmentRenderer,
                <Self as ContextHolder>::SegmentAttachedData,
            >,
            Args = (Vec<[f32; 3]>, Vec<[u32; 2]>),
        >
    {
    }
}

// Ugly, wish I had some way to avoid this
pub struct State<T>(T);

impl<T> State<T>
where
    T: private::StateTrait,
{
    fn new_inner(inner: T) -> Self {
        Self(inner)
    }

    pub fn register_surface<V: Vertices, I: Into<SurfaceIndices>>(
        &mut self,
        name: String,
        vertices: V,
        indices: I,
    ) -> SurfaceMut<T::SurfaceRenderer, T::SurfaceAttachedData, T::Context<'_>> {
        self.0.register(name, (indices.into(), vertices.into()))
    }

    pub fn get_surface_mut(
        &mut self,
        name: &str,
    ) -> Option<SurfaceMut<T::SurfaceRenderer, T::SurfaceAttachedData, T::Context<'_>>> {
        self.0.get_element_mut(name)
    }

    pub fn get_surface(
        &self,
        name: &str,
    ) -> Option<&Surface<T::SurfaceRenderer, T::SurfaceAttachedData>> {
        self.0.get_element(name)
    }

    pub fn register_point_cloud<V: Vertices>(
        &mut self,
        name: String,
        positions: V,
    ) -> PointCloudMut<T::PointCloudRenderer, T::PointCloudAttachedData, T::Context<'_>> {
        self.0.register(name, positions.into())
    }

    pub fn get_point_cloud_mut(
        &mut self,
        name: &str,
    ) -> Option<PointCloudMut<T::PointCloudRenderer, T::PointCloudAttachedData, T::Context<'_>>>
    {
        self.0.get_element_mut(name)
    }

    pub fn get_point_cloud(
        &self,
        name: &str,
    ) -> Option<&PointCloud<T::PointCloudRenderer, T::PointCloudAttachedData>> {
        self.0.get_element(name)
    }

    /// Register list of segments
    ///
    /// Arguments :
    /// * `positions`: segments extremities
    /// * `connections`: segments denoted by extremities indices
    pub fn register_segment<V: Vertices>(
        &mut self,
        name: String,
        positions: V,
        connections: Vec<[u32; 2]>,
    ) -> SegmentMut<T::SegmentRenderer, T::SegmentAttachedData, T::Context<'_>> {
        self.0.register(name, (positions.into(), connections))
    }

    pub fn get_segment_mut(
        &mut self,
        name: &str,
    ) -> Option<SegmentMut<T::SegmentRenderer, T::SegmentAttachedData, T::Context<'_>>> {
        self.0.get_element_mut(name)
    }

    pub fn get_segment(
        &self,
        name: &str,
    ) -> Option<&Segment<T::SegmentRenderer, T::SegmentAttachedData>> {
        self.0.get_element(name)
    }
}

pub use crate::point_cloud::{PointCloud, PointCloudMut};
pub use crate::segment::{Segment, SegmentMut};
pub use crate::surface::{Surface, SurfaceMut};

pub type InitialState = State<private::InnerBareState>;

/// Creates a handle to add elements to. Needs to be run after.
#[must_use]
pub fn init() -> InitialState {
    State::new_inner(private::InnerBareState {
        surfaces: IndexMap::new(),
        clouds: IndexMap::new(),
        segments: IndexMap::new(),
    })
}

impl InitialState {
    /// Show the window and start the app.
    ///
    /// In wasm, `width` and `height` are ignored and css is used to define the dimensions
    /// (allowing for dimensions in `%` and `vh`/`vw`).
    ///
    /// Arguments:
    /// * `width`: requested width of the app (no effect in wasm)
    /// * `height`: requested height of the app (no effect in wasm)
    /// * `id`: serves as window title, or id element to attach to. If `None` used `"State"`.
    /// * `settings`: global app [`Settings`]
    /// * `callback`: called every frame with a [`egui::Ui`] and a [`RunningState`] arguments, used to
    /// add UI elements and modify state accordingly.
    pub fn run<T: FnMut(&mut egui::Ui, &mut RunningState)>(
        self,
        width: u32,
        height: u32,
        id: Option<String>,
        settings: Settings,
        callback: T,
    ) {
        StateWrapper::run(self, width, height, id, settings, callback);
    }
}

pub type RunningState = State<InnerGraphicalState>;

/// Starting point to build the app.
struct StateWrapper<T: FnMut(&mut egui::Ui, &mut RunningState)> {
    init_state: Option<InitialState>,
    state: Option<RunningState>,
    ui: Option<ui::UI>,
    clipboard: Option<Clipboard>,
    id: String,
    width: u32,
    height: u32,
    proxy: EventLoopProxy<UserEvent>,
    settings: Settings,
    callback: T,
}

pub(crate) enum UserEvent {
    LoadMesh(Vec<[f32; 3]>, SurfaceIndices, String),
    Paste(String),
    Pick,
}

impl Deref for RunningState {
    type Target = private::InnerGraphicalState;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RunningState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl RunningState {
    async fn new(
        initial: InitialState,
        window: Window,
        proxy: EventLoopProxy<UserEvent>,
        settings: Settings,
    ) -> Self {
        let inner = private::InnerGraphicalState::new(initial, window, proxy, settings).await;
        Self::new_inner(inner)
    }

    /// Fit camera and ground to match the visible elements
    pub fn resize_scene(&mut self) {
        self.0.resize_scene();
    }

    /// Take a screenshot at the next frame
    pub fn screenshot(&mut self) {
        self.0.screenshot();
    }

    /// Get current selected object: first the name, then index `i` of the selected element
    ///
    /// For a surface mesh, if `i` < `nv` then the selected element si the vertex of index `i`.
    /// If `nv` <= i < `nv + nf`, it corresponds to the face of index `i - nv`.
    pub fn get_picked(&self) -> &Option<(String, usize)> {
        self.0.get_picked()
    }

    /// Politely ask to render the next frame, even if no change is detected
    pub fn refresh(&mut self) {
        self.0.refresh();
    }
}

impl private::InnerGraphicalState {
    // Initialize the state
    async fn new(
        initial: InitialState,
        window: Window,
        proxy: EventLoopProxy<UserEvent>,
        settings: Settings,
    ) -> Self {
        let private::InnerBareState {
            surfaces,
            clouds,
            segments,
        } = initial.0;
        let size = window.inner_size();
        let window = Arc::new(window);
        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
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
                        let mut limits = wgpu::Limits::downlevel_webgl2_defaults();
                        limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
                        limits.max_buffer_size = adapter.limits().max_buffer_size;
                        limits
                    } else {
                        let mut limits = wgpu::Limits::default();
                        limits.max_buffer_size = adapter.limits().max_buffer_size;
                        limits
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

        //window.request_inner_size(PhysicalSize::new(width, height));

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
        // Create depth texture
        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        // Create texture for screenshots
        let screenshoter =
            screenshot::Screenshoter::new(&device, size.width.max(1), size.height.max(1));

        let picker = picker::Picker::new(&device, size.width.max(1), size.height.max(1));

        let surfaces = surfaces
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    v.upgrade(
                        &device,
                        &camera_light_bind_group_layout,
                        &picker.bind_group_layout,
                        surface_format,
                    ),
                )
            })
            .collect();

        let clouds = clouds
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    v.upgrade(
                        &device,
                        &camera_light_bind_group_layout,
                        &picker.bind_group_layout,
                        surface_format,
                    ),
                )
            })
            .collect();
        let segments = segments
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    v.upgrade(
                        &device,
                        &camera_light_bind_group_layout,
                        &picker.bind_group_layout,
                        surface_format,
                    ),
                )
            })
            .collect();

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
        private::InnerGraphicalState {
            surfaces,
            segments,
            clouds,
            settings,
            window,
            proxy,
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
            light_uniform,
            light_buffer,
            jitter_buffer,
            camera_light_bind_group_layout,
            camera_light_bind_group,
            picker,
            //time: std::time::Instant::now(),
            dirty: true,
            egui_dirty: true,
            should_resize: true,
            copy,
            pbr_renderer,
            ground,
            taa_counter: 0,
            aabb: aabb::SBV::default(),
            rng: SmallRng::seed_from_u64(1),
        }
    }

    fn set_floor(&mut self) {
        use crate::updater::ElementGeometry;
        let mut min_y = 0.;
        for surface in self.surfaces.values() {
            if surface.shown() {
                for p in surface.geometry().get_positions() {
                    let p = glam::Mat4::from_cols_array_2d(&surface.transform.get_transform())
                        * glam::Vec3::from_array(*p).extend(1.);
                    let p = p / p[3];
                    if p[1] < min_y {
                        min_y = p[1];
                    }
                }
            }
        }
        for cloud in self.clouds.values() {
            if cloud.shown() {
                for p in cloud.geometry().get_positions() {
                    let p = glam::Mat4::from_cols_array_2d(&cloud.transform.get_transform())
                        * glam::Vec3::from_array(*p).extend(1.);
                    let p = p / p[3];
                    if p[1] < min_y {
                        min_y = p[1];
                    }
                }
            }
        }
        for segment in self.segments.values() {
            if segment.shown() {
                for p in segment.geometry().get_positions() {
                    let p = glam::Mat4::from_cols_array_2d(&segment.transform.get_transform())
                        * glam::Vec3::from_array(*p).extend(1.);
                    let p = p / p[3];
                    if p[1] < min_y {
                        min_y = p[1];
                    }
                }
            }
        }
        self.ground.set_level(&self.queue, min_y);
    }

    fn resize_scene(&mut self) {
        let mut size = None;
        let mut n = 0;
        let mut center = glam::Vec3::new(0., 0., 0.);
        for surface in self.surfaces.values() {
            if surface.shown() {
                let sbv = surface.sbv.transform(&surface.transform.get_transform());
                center += glam::Vec3::from_array(sbv.center);
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
            if cloud.shown() {
                let sbv = cloud.sbv.transform(&cloud.transform.get_transform());
                center += glam::Vec3::from_array(sbv.center);
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
        for segment in self.segments.values() {
            if segment.shown() {
                let sbv = segment.sbv.transform(&segment.transform.get_transform());
                center += glam::Vec3::from_array(sbv.center);
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
            self.screenshoter
                .resize(&self.device, new_size.width, new_size.height);
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
    fn input(&mut self, event: &WindowEvent, ui_hovered: bool) -> bool {
        // Send any input to camera controller
        let changed = self.camera_controller.process_events(event, ui_hovered);
        self.dirty |= changed;
        (!ui_hovered && self.picker.input(event)) || changed
    }

    fn update(&mut self) -> bool {
        // Sync local app state with camera
        self.camera_controller
            .update_camera(&mut self.camera, &self.settings);
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

        for surface in self.surfaces.values() {
            if surface.show {
                SBV::merge(&mut sbv, &surface.sbv);
            }
        }

        for cloud in self.clouds.values() {
            if cloud.show {
                SBV::merge(&mut sbv, &cloud.sbv);
            }
        }

        for segment in self.segments.values() {
            if segment.show {
                SBV::merge(&mut sbv, &segment.sbv);
            }
        }

        self.aabb = sbv.unwrap_or_else(aabb::SBV::default);
        if self.should_resize {
            self.resize_scene();
            self.should_resize = false;
            changed = true;
        }

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
            &self.segments,
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
                //let ampli = 0.5 + 0.5 * self.taa_counter as f32 / self.taa_frames as f32;
                let ampli = 1.;
                jitter = JitterUniform {
                    x: ampli * 2. * (self.rng.random::<f32>() - 0.5) / self.size.width as f32,
                    y: ampli * 2. * (self.rng.random::<f32>() - 0.5) / self.size.height as f32,
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
                //segment only change deph sometimes, could use conservative depth
                //surface fully uses depth buffer(except attachments)
                for cloud in self.clouds.values() {
                    cloud.render(&mut material_render_pass);
                }
                for segment in self.segments.values() {
                    segment.render(&mut material_render_pass);
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
                            r: 0.,
                            g: 0.,
                            b: 0.,
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
                if self.screenshot {
                    self.copy.screenshot(&mut render_pass);
                } else {
                    self.copy.copy(&mut render_pass);
                }

                let render_pass = render_pass.forget_lifetime();
                if !self.screenshot {
                    ui.render(render_pass, &clipped_primitives, &screen_descriptor);
                }
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
            output.unwrap().present();
        }

        self.picker.post_render(event_loop_proxy);
        if self.picker.pick_locked || !self.settings.lazy_draw {
            request_redraw = true;
        }
        Ok(request_redraw)
    }

    fn screenshot(&mut self) {
        self.screenshot = true;
    }

    fn get_picked(&self) -> &Option<(String, usize)> {
        &self.picker.picked_item
    }

    fn refresh(&mut self) {
        self.dirty = true;
    }

    #[cfg(feature = "obj_button")]
    pub(crate) fn send_mesh(&mut self, name: String) {
        let event_loop_proxy = self.proxy.clone();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let file = rfd::FileDialog::new()
                .add_filter("obj", &["obj"])
                .pick_file();
            if let Some(file_handle) = file {
                let data = file_handle;
                if let Some((mesh_v, mesh_f)) = crate::resources::load_mesh_blocking(data.into()) {
                    event_loop_proxy
                        .send_event(crate::UserEvent::LoadMesh(mesh_v, mesh_f, name))
                        .ok();
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let file = rfd::AsyncFileDialog::new()
                .add_filter("obj", &["obj"])
                .pick_file();
            let f = async move {
                let file = file.await;
                if let Some(file_handle) = file {
                    let data = file_handle.read().await;
                    if let Some((mesh_v, mesh_f)) =
                        crate::resources::parse_preloaded_mesh(data).await
                    {
                        event_loop_proxy
                            .send_event(crate::UserEvent::LoadMesh(mesh_v, mesh_f, name))
                            .ok();
                    }
                }
            };
            wasm_bindgen_futures::spawn_local(f);
        }
    }
}

impl<T: FnMut(&mut egui::Ui, &mut RunningState)> StateWrapper<T> {
    fn run(
        init_state: InitialState,
        width: u32,
        height: u32,
        id: Option<String>,
        settings: Settings,
        callback: T,
    ) {
        let id = id.unwrap_or("deuxfleurs".into());
        #[cfg(target_arch = "wasm32")]
        {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        }
        #[cfg(feature = "logger")]
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
            } else {
                env_logger::init();
            }
        }

        let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
        let proxy = event_loop.create_proxy();
        let mut app = Self {
            init_state: Some(init_state),
            state: None,
            clipboard: None,
            ui: None,
            id,
            width,
            height,
            proxy,
            settings,
            callback,
        };
        event_loop.run_app(&mut app).unwrap();
    }
}

impl<T: FnMut(&mut egui::Ui, &mut RunningState)> ApplicationHandler<UserEvent> for StateWrapper<T> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window_attributes = WindowAttributes::default()
                .with_title(&self.id)
                .with_inner_size(PhysicalSize::new(self.width, self.height));
            let window = event_loop.create_window(window_attributes).unwrap();

            #[cfg(target_arch = "wasm32")]
            {
                use winit::platform::web::WindowExtWebSys;
                web_sys::window()
                    .and_then(|win| {
                        self.clipboard = Some(win.navigator().clipboard());
                        win.document()
                    })
                    .and_then(|doc| {
                        let dst = doc.get_element_by_id(&self.id)?;
                        let canvas = window.canvas()?;
                        // disable right click
                        let empty_func = js_sys::Function::new_no_args("return false;");
                        canvas.set_oncontextmenu(Some(&empty_func));
                        dst.append_child(&canvas).ok()?;
                        Some(())
                    })
                    .expect("Couldn't append canvas to document body.");
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.clipboard = Some(Clipboard::new(
                    window.display_handle().ok().map(|handle| handle.as_raw()),
                ));
            }

            let init = self.init_state.take().unwrap();
            self.state = Some(
                RunningState::new(init, window, self.proxy.clone(), self.settings.clone())
                    .block_on(),
            );
            self.ui = Some(ui::UI::new(
                &self.state.as_ref().unwrap().device,
                event_loop,
                self.state.as_ref().unwrap().config.format,
                self.state.as_ref().unwrap().window.scale_factor(),
            ));
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        if let Some(state) = self.state.as_mut() {
            match event {
                UserEvent::LoadMesh(mesh_v, mesh_f, name) => {
                    state.register_surface(name, mesh_v, mesh_f);
                }
                UserEvent::Paste(cam) => {
                    state.camera.set(cam);
                    state.dirty = true;
                }
                UserEvent::Pick => {
                    state.0.picker.pick(
                        &state.0.surfaces,
                        &state.0.clouds,
                        &state.0.segments,
                        &state.0.camera,
                    );
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
        let (processed, ui_hovered) =
            if let (Some(ui), Some(state)) = (self.ui.as_mut(), self.state.as_mut()) {
                if window_id == state.window.id() {
                    (ui.process_event(&*state.window, &event), ui.hovered)
                } else {
                    return;
                }
            } else {
                return;
            };

        let input = if let Some(state) = self.state.as_mut() {
            state.input(&event, ui_hovered)
        } else {
            false
        };

        if processed.repaint {
            // Attempt to make egui finish animations
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
                                #[cfg(not(target_arch = "wasm32"))]
                                {
                                    let clipboard = self.clipboard.as_mut().unwrap();
                                    clipboard.set_text(cam);
                                }
                                #[cfg(target_arch = "wasm32")]
                                {
                                    let promise = self.clipboard.as_mut().unwrap().write_text(&cam);
                                    let f = async move {
                                        wasm_bindgen_futures::JsFuture::from(promise).await;
                                    };
                                    wasm_bindgen_futures::spawn_local(f);
                                }
                            }
                        } else if state.ctrl_pressed
                            && logical_key == Key::Character(SmolStr::new_inline("v"))
                            && key_state == ElementState::Pressed
                        {
                            let clipboard = self.clipboard.as_mut().unwrap();
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                if let Some(cam) = clipboard.get() {
                                    state.proxy.send_event(crate::UserEvent::Paste(cam)).ok();
                                    //state.camera.set(cam);
                                    //state.dirty = true;
                                };
                            }
                            #[cfg(target_arch = "wasm32")]
                            {
                                let promise = self.clipboard.as_mut().unwrap().read_text();
                                let event_loop_proxy = state.proxy.clone();
                                let f = async move {
                                    if let Ok(res) =
                                        wasm_bindgen_futures::JsFuture::from(promise).await
                                    {
                                        if let Some(cam) = res.as_string() {
                                            event_loop_proxy
                                                .send_event(crate::UserEvent::Paste(cam))
                                                .ok();
                                        }
                                    }
                                };
                                wasm_bindgen_futures::spawn_local(f);
                            }
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        //draw ui
                        //let mut refresh_screen = state.state.dirty;
                        ui.draw_models(
                            &*state.0.window,
                            &mut state.0.surfaces,
                            &mut state.0.clouds,
                            &mut state.0.segments,
                            state.0.camera.build_view(),
                            state.0.camera.build_proj(),
                            &state.0.device,
                            &state.0.queue,
                            &state.0.camera_light_bind_group_layout,
                            state.0.config.format,
                            &mut state.0.dirty,
                        );
                        ui.draw_callback(state, &mut self.callback);
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
                                state.0.resize(state.0.size)
                            },
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) | Err(wgpu::SurfaceError::Other) => event_loop.exit(),

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
