use crate::aabb::SBV;
use crate::attachment::{NewVectorField, VectorField};
use crate::camera::{Camera, CameraController, CameraUniform};
use crate::data::{DataSettings, DataUniformBuilder};
use crate::deferred;
use crate::picker;
use crate::point_cloud::{
    DisplayPointCloud, PointCloud, PointCloudDataBuffer, PointCloudFixedRenderer, PointCloudMut,
    PointCloudPipeline, UninitedPointCloud,
};
use crate::screenshot;
use crate::segment::{
    DisplaySegment, Segment, SegmentDataBuffer, SegmentFixedRenderer, SegmentMut, SegmentPipeline,
    UninitedSegment,
};
use crate::surface::{
    DisplaySurface, Surface, SurfaceDataBuffer, SurfaceFixedRenderer, SurfaceMut, SurfacePipeline,
    UninitedSurface,
};
use crate::texture;
use crate::types::SurfaceIndices;
use crate::types::*;
use crate::ui::UiDataElement;

use crate::Settings;
use crate::updater::{
    AttachedGeometry, DataBuffer, DisplayElement, Element, ElementGeometry, ElementMut,
    EmptyAttached, FixedRenderer, GraphicalContext, NamedSettings, NewAttachedGeometry,
    RenderPipeline, Renderer, UninitedElement,
};
use egui;
use indexmap::IndexMap;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use winit::event_loop::EventLoopProxy;
use winit::{event_loop::EventLoop, window::Window};
mod render_loop;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
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

    fn get_element_mut(&mut self, name: &str)
    -> Option<ElementMut<'_, Element, Self::Context<'_>>>;

    fn get_element(&self, name: &str) -> Option<&'_ Element>;
}

impl<Geometry, Settings, Data, Attached, T>
    GeometryHolder<UninitedElement<Geometry, Settings, Data, Attached>> for T
where
    for<'a> T: ContextHolder<Context<'a> = ()>,
    T: ContainerContextGiver<UninitedElement<Geometry, Settings, Data, Attached>>,
    Geometry: ElementGeometry,
    Settings: DataUniformBuilder + NamedSettings,
    Data: DataSettings,
    Attached: AttachedGeometry<()> + NewAttachedGeometry,
{
    type Args = Geometry::Args;

    fn register(
        &mut self,
        name: String,
        args: Self::Args,
    ) -> ElementMut<UninitedElement<Geometry, Settings, Data, Attached>, Self::Context<'_>> {
        use crate::updater::ElementTrait;
        let container = self.get_container_mut().0;
        if container.contains_key(&name) {
            let element = container.get_mut(&name).unwrap();
            element.replace(args, &mut ());
            ElementMut {
                element,
                context: (),
            }
        } else {
            let element = Element::new_bare(name.clone(), args);
            container.insert(name.clone(), element);
            ElementMut {
                element: container.get_mut(&name).unwrap(),
                context: (),
            }
        }
    }

    fn get_element_mut(
        &mut self,
        name: &str,
    ) -> Option<ElementMut<UninitedElement<Geometry, Settings, Data, Attached>, Self::Context<'_>>>
    {
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
    GeometryHolder<DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>> for T
where
    for<'a> T: ContextHolder<Context<'a> = GraphicalContext<'a>>,
    T: ContainerContextGiver<
        DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>,
    >,
    for<'a> Attached: AttachedGeometry<GraphicalContext<'a>>,
    Geometry: ElementGeometry,
    Data: DataUniformBuilder + DataSettings + UiDataElement,
    Settings: NamedSettings,
    Fixed: FixedRenderer<Geometry = Geometry>,
    DataB: DataBuffer<Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry>,
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
        use crate::updater::ElementTrait;
        let (container, mut context) = self.get_container_mut();
        // This could be better with Polonius
        if container.contains_key(&name) {
            let element = container.get_mut(&name).unwrap();
            element.replace(args, &mut context);
            ElementMut {
                element,
                context: context,
            }
        } else {
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
    ) -> Option<&DisplayElement<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>> {
        self.get_container().get(name)
    }
}

/// Holds the application state. Starting point to add visualization datas.
pub struct InnerGraphicalState {
    surfaces: IndexMap<String, DisplaySurface>,
    clouds: IndexMap<String, DisplayPointCloud>,
    segments: IndexMap<String, DisplaySegment>,
    settings: Settings,

    window: Arc<Window>,
    proxy: EventLoopProxy<UserEvent>,
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
    should_resize: bool,

    // Item picker
    picker: picker::Picker,

    copy: deferred::TextureCopy,
    pbr_renderer: deferred::PBR,
    ground: deferred::Ground,
    taa_counter: u8,
    aabb: SBV,
}

impl ContextHolder for InnerGraphicalState {
    type Context<'a> = GraphicalContext<'a>;
    type SurfaceRenderer = Renderer<SurfaceFixedRenderer, SurfaceDataBuffer, SurfacePipeline>;
    type SurfaceAttachedData = VectorField;
    type PointCloudRenderer =
        Renderer<PointCloudFixedRenderer, PointCloudDataBuffer, PointCloudPipeline>;
    type PointCloudAttachedData = EmptyAttached;
    type SegmentRenderer = Renderer<SegmentFixedRenderer, SegmentDataBuffer, SegmentPipeline>;
    type SegmentAttachedData = EmptyAttached;
}

impl ContainerContextGiver<DisplaySurface> for InnerGraphicalState {
    fn get_container(&self) -> &IndexMap<String, DisplaySurface> {
        &self.surfaces
    }

    fn get_container_mut(&mut self) -> (&mut IndexMap<String, DisplaySurface>, Self::Context<'_>) {
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

    fn get_container_mut(&mut self) -> (&mut IndexMap<String, DisplaySegment>, Self::Context<'_>) {
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
    GeometryHolder<
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

pub struct State<T>(T);

impl<T> State<T>
where
    T: StateTrait,
{
    pub(crate) fn new_inner(inner: T) -> Self {
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

pub type InitialState = State<InnerBareState>;

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
    ui: Option<crate::ui::UI>,
    id: String,
    width: u32,
    height: u32,
    proxy: EventLoopProxy<UserEvent>,
    settings: Settings,
    callback: T,
}

pub(crate) enum UserEvent {
    LoadMesh(Vec<[f32; 3]>, crate::types::SurfaceIndices, String),
    Pick,
}

impl Deref for RunningState {
    type Target = InnerGraphicalState;

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
        let inner = InnerGraphicalState::new(initial, window, proxy, settings).await;
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
            init_state: Some(init_state),
            state: None,
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
