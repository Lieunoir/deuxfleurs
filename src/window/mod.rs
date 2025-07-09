use crate::aabb::SBV;
use crate::camera::{Camera, CameraController, CameraUniform};
use crate::data::internal::{DataSettings, DataUniformBuilder};
use crate::deferred;
use crate::picker::{self, Picked};
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
    DisplaySurface, NewSurfaceAttachment, Surface, SurfaceAttachment, SurfaceDataBuffer,
    SurfaceFixedRenderer, SurfaceMut, SurfacePipeline, UninitedSurface,
};
use crate::texture;
use crate::types::SurfaceIndices;
use crate::types::*;
use crate::ui::UiDataElement;
#[cfg(not(target_arch = "wasm32"))]
use egui_winit::clipboard::Clipboard;
use pollster::FutureExt;
use serde::{Deserialize, Serialize};
#[cfg(target_arch = "wasm32")]
use web_sys::Clipboard;

use crate::Settings;
use crate::shape::{
    AttachedGeometry, DataBuffer, DisplayShape, EmptyAttached, FixedRenderer, GraphicalContext,
    NewAttachedGeometry, Render, RenderPipeline, Renderer, Shape, ShapeGeometry, ShapeMut,
    ShapeSettings, UninitedShape,
};
use egui;
use indexmap::IndexMap;
use rand::rngs::SmallRng;
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

pub trait ContainerContextGiver<Shape>: ContextHolder {
    fn get_container_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, Shape>,
        Self::Context<'_>,
        Option<&mut bool>,
        Option<&mut bool>,
        Option<&mut Option<(String, Picked)>>,
    );

    fn get_container(&self) -> &IndexMap<String, Shape>;
}

pub trait GeometryHolder<Shape>: ContainerContextGiver<Shape> {
    type Args;

    fn register(
        &mut self,
        name: String,
        args: Self::Args,
    ) -> ShapeMut<'_, Shape, Self::Context<'_>>;

    fn get_shape_mut(&mut self, name: &str) -> Option<ShapeMut<'_, Shape, Self::Context<'_>>>;

    fn get_shape(&self, name: &str) -> Option<&'_ Shape>;

    fn remove_shape(&mut self, name: &str);
}

impl<Geometry, Settings, Data, Attached, T>
    GeometryHolder<UninitedShape<Geometry, Settings, Data, Attached>> for T
where
    for<'a> T: ContextHolder<Context<'a> = &'a mut crate::Settings>,
    T: ContainerContextGiver<UninitedShape<Geometry, Settings, Data, Attached>>,
    Geometry: ShapeGeometry,
    Settings: DataUniformBuilder + ShapeSettings,
    Data: DataSettings,
    for<'a> Attached: AttachedGeometry<&'a mut crate::Settings> + NewAttachedGeometry,
{
    type Args = Geometry::Args;

    fn register(
        &mut self,
        name: String,
        args: Self::Args,
    ) -> ShapeMut<UninitedShape<Geometry, Settings, Data, Attached>, Self::Context<'_>> {
        use crate::shape::ShapeTrait;
        let (container, mut context, _, _, _) = self.get_container_mut();
        if container.contains_key(&name) {
            let shape = container.get_mut(&name).unwrap();
            shape.replace(args, &mut context);
            ShapeMut {
                inner: shape,
                context,
            }
        } else {
            let shape = Shape::new_bare(name.clone(), args, None);
            container.insert(name.clone(), shape);
            ShapeMut {
                inner: container.get_mut(&name).unwrap(),
                context,
            }
        }
    }

    fn get_shape_mut(
        &mut self,
        name: &str,
    ) -> Option<ShapeMut<UninitedShape<Geometry, Settings, Data, Attached>, Self::Context<'_>>>
    {
        let (container, context, _, _, _) = self.get_container_mut();
        container.get_mut(name).map(|shape| ShapeMut {
            inner: shape,
            context,
        })
    }

    fn get_shape(
        &self,
        name: &str,
    ) -> Option<&'_ UninitedShape<Geometry, Settings, Data, Attached>> {
        self.get_container().get(name)
    }

    fn remove_shape(&mut self, name: &str) {
        let (container, _context, _should_resize, _counters_dirty, _picked) =
            self.get_container_mut();
        container.shift_remove(name);
    }
}

impl<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached, T>
    GeometryHolder<DisplayShape<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>> for T
where
    for<'a> T: ContextHolder<Context<'a> = GraphicalContext<'a>>,
    T: ContainerContextGiver<
        DisplayShape<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>,
    >,
    for<'a> Attached: AttachedGeometry<GraphicalContext<'a>>,
    Geometry: ShapeGeometry + Clone,
    Data: DataUniformBuilder + DataSettings + UiDataElement + Clone,
    Settings: ShapeSettings + Clone,
    Fixed: FixedRenderer<Geometry = Geometry>,
    DataB: DataBuffer<Data = Data, Geometry = Geometry>,
    Pipeline: RenderPipeline<Settings = Settings, Data = Data, Geometry = Geometry>,
    Renderer<Fixed, DataB, Pipeline>: Render,
{
    type Args = Geometry::Args;

    fn register(
        &mut self,
        name: String,
        args: Self::Args,
    ) -> ShapeMut<
        DisplayShape<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>,
        Self::Context<'_>,
    > {
        use crate::shape::ShapeTrait;
        let (container, mut context, should_resize, counters_dirty, picked) =
            self.get_container_mut();
        *context.refresh_screen = true;
        // This could be better with Polonius
        if container.contains_key(&name) {
            let shape = container.get_mut(&name).unwrap();
            if !shape.replace(args, &mut context) {
                should_resize.map(|should_resize| *should_resize = true);
                counters_dirty.map(|counters_dirty| *counters_dirty = true);
                let picked = picked.unwrap();
                if let Some((picked_name, _picked)) = picked {
                    if *picked_name == name {
                        *picked = None;
                    }
                }
            }
            ShapeMut {
                inner: shape,
                context: context,
            }
        } else {
            let shape = Shape::new(
                name.clone(),
                args,
                None,
                context.device,
                context.camera_light_bind_group_layout,
                context.counter_bind_group_layout,
                context.color_format,
            );
            container.insert(name.clone(), shape);
            should_resize.map(|should_resize| *should_resize = true);
            counters_dirty.map(|counters_dirty| *counters_dirty = true);
            let picked = picked.unwrap();
            if let Some((picked_name, _picked)) = picked {
                if *picked_name == name {
                    *picked = None;
                }
            }
            ShapeMut {
                inner: container.get_mut(&name).unwrap(),
                context,
            }
        }
    }

    fn get_shape_mut(
        &mut self,
        name: &str,
    ) -> Option<
        ShapeMut<
            DisplayShape<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>,
            Self::Context<'_>,
        >,
    > {
        let (container, context, _, _, _) = self.get_container_mut();
        container.get_mut(name).map(|shape| ShapeMut {
            inner: shape,
            context,
        })
    }

    fn get_shape(
        &self,
        name: &str,
    ) -> Option<&DisplayShape<Geometry, Fixed, DataB, Pipeline, Settings, Data, Attached>> {
        self.get_container().get(name)
    }

    fn remove_shape(&mut self, name: &str) {
        let (container, context, _should_resize, counters_dirty, picked) = self.get_container_mut();
        if let Some(shape) = container.shift_remove(name) {
            let picked = picked.unwrap();
            if let Some((picked_name, _picked)) = picked {
                if *picked_name == name {
                    *picked = None;
                }
            }
            *context.refresh_screen |= shape.show;
            *counters_dirty.unwrap() = true;
        }
    }
}

/// Holds the application state. Starting point to add visualization datas.
pub struct InnerGraphicalState {
    surfaces: IndexMap<String, DisplaySurface>,
    clouds: IndexMap<String, DisplayPointCloud>,
    segments: IndexMap<String, DisplaySegment>,
    pub(crate) settings: Settings,

    window: Option<Arc<Window>>,
    proxy: Option<EventLoopProxy<UserEvent>>,
    // Graphic context
    surface: Option<wgpu::Surface<'static>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    // Window size
    size: winit::dpi::PhysicalSize<u32>,
    // Textures
    depth_texture: texture::Texture,
    // Screenshots
    screenshoter: screenshot::Screenshoter,

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
    pub(crate) dirty: bool,
    egui_dirty: bool,
    should_resize: bool,

    // Item picker
    pub(crate) picker: picker::Picker,

    copy: deferred::TextureCopy,
    pbr_renderer: deferred::PBR,
    ground: deferred::Ground,
    taa_counter: u8,
    aabb: SBV,
    rng: SmallRng,
}

impl ContextHolder for InnerGraphicalState {
    type Context<'a> = GraphicalContext<'a>;
    type SurfaceRenderer = Renderer<SurfaceFixedRenderer, SurfaceDataBuffer, SurfacePipeline>;
    type SurfaceAttachedData = SurfaceAttachment;
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

    fn get_container_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, DisplaySurface>,
        Self::Context<'_>,
        Option<&mut bool>,
        Option<&mut bool>,
        Option<&mut Option<(String, Picked)>>,
    ) {
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
            Some(&mut self.should_resize),
            Some(&mut self.picker.counters_dirty),
            Some(&mut self.picker.picked_item),
        )
    }
}

impl ContainerContextGiver<DisplayPointCloud> for InnerGraphicalState {
    fn get_container(&self) -> &IndexMap<String, DisplayPointCloud> {
        &self.clouds
    }

    fn get_container_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, DisplayPointCloud>,
        Self::Context<'_>,
        Option<&mut bool>,
        Option<&mut bool>,
        Option<&mut Option<(String, Picked)>>,
    ) {
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
            Some(&mut self.should_resize),
            Some(&mut self.picker.counters_dirty),
            Some(&mut self.picker.picked_item),
        )
    }
}

impl ContainerContextGiver<DisplaySegment> for InnerGraphicalState {
    fn get_container(&self) -> &IndexMap<String, DisplaySegment> {
        &self.segments
    }

    fn get_container_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, DisplaySegment>,
        Self::Context<'_>,
        Option<&mut bool>,
        Option<&mut bool>,
        Option<&mut Option<(String, Picked)>>,
    ) {
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
            Some(&mut self.should_resize),
            Some(&mut self.picker.counters_dirty),
            Some(&mut self.picker.picked_item),
        )
    }
}

impl StateTrait for InnerGraphicalState {}

pub struct InnerBareState<T: FnMut(&mut egui::Ui, &mut RunningState)> {
    pub(crate) surfaces: IndexMap<String, UninitedSurface>,
    pub(crate) clouds: IndexMap<String, UninitedPointCloud>,
    pub(crate) segments: IndexMap<String, UninitedSegment>,
    pub settings: Settings,
    pub camera: Camera,
    pub(crate) callback: T,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct InnerBareStateSerde {
    pub(crate) surfaces: IndexMap<String, UninitedSurface>,
    pub(crate) clouds: IndexMap<String, UninitedPointCloud>,
    pub(crate) segments: IndexMap<String, UninitedSegment>,
    pub settings: Settings,
    pub camera: Camera,
}

impl<T: FnMut(&mut egui::Ui, &mut RunningState)> ContextHolder for InnerBareState<T> {
    type Context<'a> = &'a mut Settings;
    type SurfaceRenderer = ();
    type SurfaceAttachedData = NewSurfaceAttachment;
    type PointCloudRenderer = ();
    type PointCloudAttachedData = ();
    type SegmentRenderer = ();
    type SegmentAttachedData = ();
}

impl<T: FnMut(&mut egui::Ui, &mut RunningState)> ContainerContextGiver<UninitedSurface>
    for InnerBareState<T>
{
    fn get_container(&self) -> &IndexMap<String, UninitedSurface> {
        &self.surfaces
    }

    fn get_container_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, UninitedSurface>,
        &mut Settings,
        Option<&mut bool>,
        Option<&mut bool>,
        Option<&mut Option<(String, Picked)>>,
    ) {
        (&mut self.surfaces, &mut self.settings, None, None, None)
    }
}

impl<T: FnMut(&mut egui::Ui, &mut RunningState)> ContainerContextGiver<UninitedPointCloud>
    for InnerBareState<T>
{
    fn get_container(&self) -> &IndexMap<String, UninitedPointCloud> {
        &self.clouds
    }

    fn get_container_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, UninitedPointCloud>,
        &mut Settings,
        Option<&mut bool>,
        Option<&mut bool>,
        Option<&mut Option<(String, Picked)>>,
    ) {
        (&mut self.clouds, &mut self.settings, None, None, None)
    }
}

impl<T: FnMut(&mut egui::Ui, &mut RunningState)> ContainerContextGiver<UninitedSegment>
    for InnerBareState<T>
{
    fn get_container(&self) -> &IndexMap<String, UninitedSegment> {
        &self.segments
    }

    fn get_container_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, UninitedSegment>,
        &mut Settings,
        Option<&mut bool>,
        Option<&mut bool>,
        Option<&mut Option<(String, Picked)>>,
    ) {
        (&mut self.segments, &mut self.settings, None, None, None)
    }
}

impl<T: FnMut(&mut egui::Ui, &mut RunningState)> StateTrait for InnerBareState<T> {}

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

pub struct State<T>(pub(crate) T);

impl<T: StateTrait> State<T> {
    pub(crate) fn new_inner(inner: T) -> Self {
        Self(inner)
    }

    pub fn register_surface<V: Vertices, I: Into<SurfaceIndices>>(
        &mut self,
        name: impl Into<String>,
        vertices: V,
        indices: I,
    ) -> SurfaceMut<T::SurfaceRenderer, T::SurfaceAttachedData, T::Context<'_>> {
        self.0
            .register(name.into(), (indices.into(), vertices.into()))
    }

    pub fn get_surface_mut(
        &mut self,
        name: &str,
    ) -> Option<SurfaceMut<T::SurfaceRenderer, T::SurfaceAttachedData, T::Context<'_>>> {
        self.0.get_shape_mut(name)
    }

    pub fn get_surface(
        &self,
        name: &str,
    ) -> Option<&Surface<T::SurfaceRenderer, T::SurfaceAttachedData>> {
        self.0.get_shape(name)
    }

    pub fn remove_surface(&mut self, name: &str) {
        <T as GeometryHolder<Surface<T::SurfaceRenderer, T::SurfaceAttachedData>>>::remove_shape(
            &mut self.0,
            name,
        );
    }

    pub fn register_point_cloud<V: Vertices>(
        &mut self,
        name: impl Into<String>,
        positions: V,
    ) -> PointCloudMut<T::PointCloudRenderer, T::PointCloudAttachedData, T::Context<'_>> {
        self.0.register(name.into(), positions.into())
    }

    pub fn get_point_cloud_mut(
        &mut self,
        name: &str,
    ) -> Option<PointCloudMut<T::PointCloudRenderer, T::PointCloudAttachedData, T::Context<'_>>>
    {
        self.0.get_shape_mut(name)
    }

    pub fn get_point_cloud(
        &self,
        name: &str,
    ) -> Option<&PointCloud<T::PointCloudRenderer, T::PointCloudAttachedData>> {
        self.0.get_shape(name)
    }

    pub fn remove_point_cloud(&mut self, name: &str) {
        <T as GeometryHolder<PointCloud<T::PointCloudRenderer, T::PointCloudAttachedData>>>::remove_shape(
                &mut self.0,
                name,
            );
    }

    /// Register list of segments
    ///
    /// Arguments :
    /// * `positions`: segments extremities
    /// * `connections`: segments denoted by extremities indices
    pub fn register_segment<V: Vertices>(
        &mut self,
        name: impl Into<String>,
        positions: V,
        connections: Vec<[u32; 2]>,
    ) -> SegmentMut<T::SegmentRenderer, T::SegmentAttachedData, T::Context<'_>> {
        self.0
            .register(name.into(), (positions.into(), connections))
    }

    pub fn get_segment_mut(
        &mut self,
        name: &str,
    ) -> Option<SegmentMut<T::SegmentRenderer, T::SegmentAttachedData, T::Context<'_>>> {
        self.0.get_shape_mut(name)
    }

    pub fn get_segment(
        &self,
        name: &str,
    ) -> Option<&Segment<T::SegmentRenderer, T::SegmentAttachedData>> {
        self.0.get_shape(name)
    }

    pub fn remove_segment(&mut self, name: &str) {
        <T as GeometryHolder<Segment<T::SegmentRenderer, T::SegmentAttachedData>>>::remove_shape(
            &mut self.0,
            name,
        );
    }
}

/// Starting point to build the app.
pub type InitialState<T: FnMut(&mut egui::Ui, &mut RunningState)> = State<InnerBareState<T>>;

impl<T: FnMut(&mut egui::Ui, &mut RunningState)> InitialState<T> {
    /// Show the window and start the app.
    ///
    /// In wasm, `width` and `height` are ignored and css is used to define the dimensions
    /// (allowing for dimensions in `%` and `vh`/`vw`).
    ///
    /// Arguments:
    /// * `width`: requested width of the app (no effect in wasm)
    /// * `height`: requested height of the app (no effect in wasm)
    /// * `id`: serves as window title, or id shape to attach to. If `None` uses `"State"`.
    ///
    /// ```
    /// use deuxfleurs::{Settings, load_mesh};
    ///
    /// # fn main() {
    /// #     pollster::block_on(run());
    /// # }
    /// # pub async fn run() {
    /// let (spot_v, spot_f) = load_mesh("examples/assets/spot.obj").await.unwrap();
    /// let mut handle = deuxfleurs::init(Settings::default());
    /// handle.register_surface("Spot", spot_v, spot_f);
    /// let mut handle = handle.run(1920, 1080, Some("deuxfleurs"));
    /// # }
    /// ```
    pub fn run<S: Into<String>>(self, width: u32, height: u32, id: Option<S>) {
        StateWrapper::run(self, width, height, id.map(Into::into));
    }

    /// Run the app without a window. Allows running the app in environment where no
    /// display is available and taking screenshots automatically.
    ///
    /// Currently only available on non wasm targets, as webGL requires a context.
    ///
    /// ```
    /// use deuxfleurs::{Settings, load_mesh};
    ///
    /// # fn main() {
    /// #     pollster::block_on(run());
    /// # }
    /// # pub async fn run() {
    /// let (spot_v, spot_f) = load_mesh("examples/assets/spot.obj").await.unwrap();
    /// let mut handle = deuxfleurs::init(Settings::default());
    /// handle.register_surface("Spot", spot_v, spot_f);
    /// let mut handle = handle.run_headless();
    /// handle.screenshot();
    /// # }
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg_attr(docsrs, doc(cfg(not(target_arch = "wasm32"))))]
    pub fn run_headless(self) -> RunningState {
        let inner = InnerGraphicalState::new(
            self.0.surfaces,
            self.0.clouds,
            self.0.segments,
            self.0.settings,
            self.0.camera,
            None,
            None,
        )
        .block_on();
        RunningState::new_inner(inner)
    }

    /// Specify a callback that will be called once every frame.
    ///
    /// Passes an [`egui::Ui`] and a [`RunningState`] arguments which can be
    /// used to add UI elements and modify state accordingly.
    pub fn with_callback<U: FnMut(&mut egui::Ui, &mut RunningState)>(
        self,
        callback: U,
    ) -> InitialState<U> {
        let InnerBareState {
            surfaces,
            clouds,
            segments,
            settings,
            camera,
            ..
        } = self.0;
        let inner = InnerBareState {
            surfaces,
            clouds,
            segments,
            settings,
            callback,
            camera,
        };
        InitialState::new_inner(inner)
    }
}

/// Holds the application state. Starting point to add visualization datas.
pub type RunningState = State<InnerGraphicalState>;

struct StateWrapper<T: FnMut(&mut egui::Ui, &mut RunningState)> {
    init_state: Option<InitialState<T>>,
    state: Option<RunningState>,
    ui: Option<crate::ui::UI>,
    clipboard: Option<Clipboard>,
    callback: Option<T>,
    id: String,
    width: u32,
    height: u32,
    proxy: EventLoopProxy<UserEvent>,
}

pub(crate) enum UserEvent {
    #[cfg(feature = "obj_button")]
    LoadMesh(Vec<[f32; 3]>, crate::types::SurfaceIndices, String),
    Paste(String),
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
        surfaces: IndexMap<String, UninitedSurface>,
        clouds: IndexMap<String, UninitedPointCloud>,
        segments: IndexMap<String, UninitedSegment>,
        camera: Camera,
        settings: Settings,
        window: Window,
        proxy: EventLoopProxy<UserEvent>,
    ) -> Self {
        let inner = InnerGraphicalState::new(
            surfaces,
            clouds,
            segments,
            settings,
            camera,
            Some(window),
            Some(proxy),
        )
        .await;
        Self::new_inner(inner)
    }

    /// Fit camera and ground to match the visible shapes
    pub fn resize_scene(&mut self) {
        self.0.resize_scene();
    }

    /// Take a screenshot of the scene.
    pub fn screenshot(&mut self) {
        self.0.screenshot();
    }

    /// Get current selected object: first the name, then index `i` and type of the selected element
    pub fn get_picked(&self) -> &Option<(String, Picked)> {
        self.0.get_picked()
    }

    /// Politely ask to render the next frame, even if no change is detected
    pub fn refresh(&mut self) {
        self.0.refresh();
    }
}

impl<T: FnMut(&mut egui::Ui, &mut RunningState)> StateWrapper<T> {
    fn run(init_state: InitialState<T>, width: u32, height: u32, id: Option<String>) {
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
            callback: None,
            ui: None,
            id,
            width,
            height,
            proxy,
        };
        event_loop.run_app(&mut app).unwrap();
    }
}
