use crate::Settings;
use crate::camera::{Camera, CameraController, CameraUniform};
use crate::data::internal::DataUniform;
use crate::picker::{self, Picked};
use crate::point_cloud::geometry::PointCloudDesc;
use crate::point_cloud::{DisplayPointCloud, PointCloud, PointCloudMut, UninitedPointCloud};
use crate::post_process;
use crate::sbv::SBV;
use crate::screenshot;
use crate::segment::geometry::SegmentDesc;
use crate::segment::{DisplaySegment, Segment, SegmentMut, UninitedSegment};
use crate::shape::{
    DataBuffer, DisplayShape, FixedRenderer, GraphicalContext, InvariantShapeDescriptor,
    NewAttachedGeometry, Render, RenderPipeline, Renderer, Shape, ShapeDescriptor, ShapeGeometry,
    ShapeMut, UninitedShape,
};
use crate::surface::geometry::SurfaceDesc;
use crate::surface::{DisplaySurface, Surface, SurfaceMut, UninitedSurface};
use crate::texture::TextureBufferPool;
use crate::types::SurfaceIndices;
use crate::types::*;
use egui;
#[cfg(not(target_arch = "wasm32"))]
use egui_winit::clipboard::Clipboard;
use indexmap::IndexMap;
use pollster::FutureExt;
use rand::rngs::SmallRng;
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::Clipboard;
use wgpu_profiler::GpuProfiler;
use winit::event_loop::EventLoopProxy;
use winit::{event_loop::EventLoop, window::Window};
mod render_loop;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(module = "/src/save.js")]
extern "C" {
    fn save_state(filename: &str, data: &[u8]);
}

pub trait ContextHolder {
    type Context<'a>;
    type ExtendedContext<'a>;
    type DataUniform<'a>;
    type TransformLayout;

    fn get_settings<'a>(ctxt: &'a Self::Context<'_>) -> &'a Settings;

    fn reborrow_context<'a: 'b, 'b>(ctxt: &'b mut Self::Context<'a>) -> Self::Context<'b>;
}

//Can't be merged with above type due to https://github.com/rust-lang/rust/issues/87479
pub trait ContainersHolder: ContextHolder
where
    SurfaceDesc: ShapeDescriptor<Self>,
    PointCloudDesc: ShapeDescriptor<Self>,
    SegmentDesc: ShapeDescriptor<Self>,
{
    fn get_containers_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, Surface<Self>>,
        &mut IndexMap<String, PointCloud<Self>>,
        &mut IndexMap<String, Segment<Self>>,
        Self::Context<'_>,
        Self::ExtendedContext<'_>,
    );

    fn get_containers(
        &self,
    ) -> (
        &IndexMap<String, Surface<Self>>,
        &IndexMap<String, PointCloud<Self>>,
        &IndexMap<String, Segment<Self>>,
    );
}

pub trait ContainerContextGiver<Shape>: ContainersHolder
where
    SurfaceDesc: ShapeDescriptor<Self>,
    PointCloudDesc: ShapeDescriptor<Self>,
    SegmentDesc: ShapeDescriptor<Self>,
{
    fn get_container_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, Shape>,
        Self::Context<'_>,
        Self::ExtendedContext<'_>,
    );

    fn get_container(&self) -> &IndexMap<String, Shape>;
}

impl<State: ContainersHolder> ContainerContextGiver<Surface<State>> for State
where
    SurfaceDesc: ShapeDescriptor<State>,
    PointCloudDesc: ShapeDescriptor<State>,
    SegmentDesc: ShapeDescriptor<State>,
{
    fn get_container_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, Surface<State>>,
        Self::Context<'_>,
        Self::ExtendedContext<'_>,
    ) {
        let (surf, _pc, _seg, ctxt, ectxt) = self.get_containers_mut();
        (surf, ctxt, ectxt)
    }

    fn get_container(&self) -> &IndexMap<String, Surface<State>> {
        self.get_containers().0
    }
}

impl<State: ContainersHolder> ContainerContextGiver<PointCloud<State>> for State
where
    SurfaceDesc: ShapeDescriptor<State>,
    PointCloudDesc: ShapeDescriptor<State>,
    SegmentDesc: ShapeDescriptor<State>,
{
    fn get_container_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, PointCloud<State>>,
        Self::Context<'_>,
        Self::ExtendedContext<'_>,
    ) {
        let (_surf, pc, _seg, ctxt, ectxt) = self.get_containers_mut();
        (pc, ctxt, ectxt)
    }

    fn get_container(&self) -> &IndexMap<String, PointCloud<State>> {
        self.get_containers().1
    }
}

impl<State: ContainersHolder> ContainerContextGiver<Segment<State>> for State
where
    SurfaceDesc: ShapeDescriptor<State>,
    PointCloudDesc: ShapeDescriptor<State>,
    SegmentDesc: ShapeDescriptor<State>,
{
    fn get_container_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, Segment<State>>,
        Self::Context<'_>,
        Self::ExtendedContext<'_>,
    ) {
        let (_surf, _pc, seg, ctxt, ectxt) = self.get_containers_mut();
        (seg, ctxt, ectxt)
    }

    fn get_container(&self) -> &IndexMap<String, Segment<State>> {
        self.get_containers().2
    }
}

pub trait GeometryHolder<Shape>: ContainerContextGiver<Shape> + Sized
where
    SurfaceDesc: ShapeDescriptor<Self>,
    PointCloudDesc: ShapeDescriptor<Self>,
    SegmentDesc: ShapeDescriptor<Self>,
{
    type Args;

    fn register(&mut self, name: String, args: Self::Args) -> ShapeMut<'_, Shape, Self>;

    fn get_shape_mut(&mut self, name: &str) -> Option<ShapeMut<'_, Shape, Self>>;

    fn get_shape(&self, name: &str) -> Option<&'_ Shape>;

    fn remove_shape(&mut self, name: &str);
}

impl<Desc> GeometryHolder<UninitedShape<Desc>> for InnerBareState
where
    Desc: ShapeDescriptor<InnerBareState, Renderer = ()>,
    Desc::AttachedGeometry: NewAttachedGeometry,
    InnerBareState: ContainerContextGiver<UninitedShape<Desc>>,
{
    type Args = <<Desc as InvariantShapeDescriptor>::Geometry as ShapeGeometry>::Args;

    fn register(
        &mut self,
        name: String,
        args: Self::Args,
    ) -> ShapeMut<'_, UninitedShape<Desc>, Self> {
        use crate::shape::ShapeTrait;
        let (container, mut context, _) = self.get_container_mut();
        if container.contains_key(&name) {
            let shape = container.get_mut(&name).unwrap();
            Desc::replace(shape, args, &mut context);
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

    fn get_shape_mut(&mut self, name: &str) -> Option<ShapeMut<'_, UninitedShape<Desc>, Self>> {
        let (container, context, _) = self.get_container_mut();
        container.get_mut(name).map(|shape| ShapeMut {
            inner: shape,
            context,
        })
    }

    fn get_shape(&self, name: &str) -> Option<&'_ UninitedShape<Desc>> {
        self.get_container().get(name)
    }

    fn remove_shape(&mut self, name: &str) {
        self.get_container_mut().0.shift_remove(name);
    }
}

impl<Desc, Fixed, DataB, Pipeline> GeometryHolder<DisplayShape<Desc>> for InnerGraphicalState
where
    Desc: ShapeDescriptor<InnerGraphicalState, Renderer = Renderer<Fixed, DataB, Pipeline>>,
    Fixed: FixedRenderer<Geometry = Desc::Geometry>,
    DataB: DataBuffer<Data = Desc::Data, Geometry = Desc::Geometry>,
    Pipeline:
        RenderPipeline<Settings = Desc::Settings, Data = Desc::Data, Geometry = Desc::Geometry>,
    Renderer<Fixed, DataB, Pipeline>: Render,
    InnerGraphicalState: ContainerContextGiver<DisplayShape<Desc>>,
{
    type Args = <<Desc as InvariantShapeDescriptor>::Geometry as ShapeGeometry>::Args;

    fn register(
        &mut self,
        name: String,
        args: Self::Args,
    ) -> ShapeMut<'_, DisplayShape<Desc>, InnerGraphicalState> {
        use crate::shape::ShapeTrait;
        let (container, mut context, (should_resize, counters_dirty, picked)) =
            self.get_container_mut();
        *context.refresh_screen = true;
        // This could be better with Polonius
        if container.contains_key(&name) {
            let shape = container.get_mut(&name).unwrap();
            if !Desc::replace(shape, args, &mut context) {
                *should_resize = true;
                *counters_dirty = true;
                if let Some((picked_name, _picked)) = picked
                    && *picked_name == name
                {
                    *picked = None;
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
                context.camera_bind_group_layout,
                context.counter_bind_group_layout,
            );
            container.insert(name.clone(), shape);
            *should_resize = true;
            *counters_dirty = true;
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
    ) -> Option<ShapeMut<'_, DisplayShape<Desc>, InnerGraphicalState>> {
        let (container, context, _) = self.get_container_mut();
        container.get_mut(name).map(|shape| ShapeMut {
            inner: shape,
            context,
        })
    }

    fn get_shape(&self, name: &str) -> Option<&DisplayShape<Desc>> {
        self.get_container().get(name)
    }

    fn remove_shape(&mut self, name: &str) {
        let (container, context, (_should_resize, counters_dirty, picked)) =
            self.get_container_mut();
        if let Some(shape) = container.shift_remove(name) {
            if let Some((picked_name, _picked)) = picked {
                if *picked_name == name {
                    *picked = None;
                }
            }
            *context.refresh_screen |= shape.show;
            *counters_dirty = true;
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct JitterUniform {
    x: f32,
    y: f32,
    _padding: [u32; 2],
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
    texture_buffer_pool: TextureBufferPool,
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
    jitter_buffer: wgpu::Buffer,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_bind_group: wgpu::BindGroup,
    // egui
    //ui: ui::UI,
    //time: std::time::Instant,
    pub(crate) dirty: bool,
    egui_dirty: bool,
    should_resize: bool,

    // Item picker
    pub(crate) picker: picker::Picker,

    copy: post_process::TextureCopy,
    pbr_renderer: post_process::PBR,
    ground: post_process::Ground,
    ssao: post_process::SSAO,
    taa_counter: u8,
    sbv: SBV,
    rng: SmallRng,

    profiler: GpuProfiler,
}

impl ContextHolder for InnerGraphicalState {
    type Context<'a> = GraphicalContext<'a>;
    type ExtendedContext<'a> = (&'a mut bool, &'a mut bool, &'a mut Option<(String, Picked)>);
    type DataUniform<'a> = &'a Option<DataUniform>;
    type TransformLayout = wgpu::BindGroupLayout;

    fn get_settings<'a>(ctxt: &'a Self::Context<'_>) -> &'a Settings {
        ctxt.settings
    }

    fn reborrow_context<'a: 'b, 'b>(ctxt: &'b mut Self::Context<'a>) -> Self::Context<'b> {
        GraphicalContext {
            settings: ctxt.settings,
            device: ctxt.device,
            queue: ctxt.queue,
            camera_bind_group_layout: ctxt.camera_bind_group_layout,
            counter_bind_group_layout: ctxt.counter_bind_group_layout,
            refresh_screen: ctxt.refresh_screen,
        }
    }
}

impl ContainersHolder for InnerGraphicalState {
    fn get_containers(
        &self,
    ) -> (
        &IndexMap<String, Surface<Self>>,
        &IndexMap<String, PointCloud<Self>>,
        &IndexMap<String, Segment<Self>>,
    ) {
        (&self.surfaces, &self.clouds, &self.segments)
    }

    fn get_containers_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, Surface<Self>>,
        &mut IndexMap<String, PointCloud<Self>>,
        &mut IndexMap<String, Segment<Self>>,
        Self::Context<'_>,
        Self::ExtendedContext<'_>,
    ) {
        (
            &mut self.surfaces,
            &mut self.clouds,
            &mut self.segments,
            Self::Context {
                settings: &self.settings,
                device: &self.device,
                queue: &self.queue,
                camera_bind_group_layout: &self.camera_bind_group_layout,
                counter_bind_group_layout: &self.picker.bind_group_layout,
                refresh_screen: &mut self.dirty,
            },
            (
                &mut self.should_resize,
                &mut self.picker.counters_dirty,
                &mut self.picker.picked_item,
            ),
        )
    }
}

impl StateTrait for InnerGraphicalState {
    #[cfg(feature = "saves")]
    fn load_from_state_slice(&mut self, data: &[u8]) -> Result<(), ()> {
        let bared = serde_cbor::from_slice(data).map_err(|_| ())?;
        self.receive_save(bared);
        Ok(())
    }

    #[cfg(feature = "saves")]
    fn save_state_vec(&self) -> Result<Vec<u8>, ()> {
        let surfaces = self
            .surfaces
            .iter()
            .map(|(name, field)| (name.clone(), field.downgrade()))
            .collect();
        let clouds = self
            .clouds
            .iter()
            .map(|(name, field)| (name.clone(), field.downgrade()))
            .collect();
        let segments = self
            .segments
            .iter()
            .map(|(name, field)| (name.clone(), field.downgrade()))
            .collect();
        let bared = InnerBareStateSerde {
            settings: self.settings.clone(),
            camera: self.camera.clone(),
            surfaces,
            clouds,
            segments,
            ground_level: self.ground.level,
        };
        serde_cbor::to_vec(&bared).map_err(|_| ())
    }

    fn get_camera(&self) -> &Camera {
        &self.camera
    }

    fn get_camera_mut(&mut self) -> &mut Camera {
        // Conservative
        self.dirty = true;
        &mut self.camera
    }

    fn get_settings(&self) -> &Settings {
        &self.settings
    }

    fn get_settings_mut(&mut self) -> &mut Settings {
        // Conservative
        self.dirty = true;
        &mut self.settings
    }
}

pub struct InnerBareState {
    pub(crate) surfaces: IndexMap<String, UninitedSurface>,
    pub(crate) clouds: IndexMap<String, UninitedPointCloud>,
    pub(crate) segments: IndexMap<String, UninitedSegment>,
    pub settings: Settings,
    pub camera: Camera,
}

#[cfg(feature = "saves")]
#[derive(Serialize, Deserialize)]
pub(crate) struct InnerBareStateSerde {
    pub(crate) surfaces: IndexMap<String, UninitedSurface>,
    pub(crate) clouds: IndexMap<String, UninitedPointCloud>,
    pub(crate) segments: IndexMap<String, UninitedSegment>,
    pub(crate) settings: Settings,
    pub(crate) camera: Camera,
    pub(crate) ground_level: f32,
}

impl ContextHolder for InnerBareState {
    type Context<'a> = &'a mut Settings;
    type ExtendedContext<'a> = ();
    type DataUniform<'a> = ();
    type TransformLayout = ();

    fn get_settings<'a>(ctxt: &'a Self::Context<'_>) -> &'a Settings {
        ctxt
    }

    fn reborrow_context<'a: 'b, 'b>(ctxt: &'b mut Self::Context<'a>) -> Self::Context<'b> {
        ctxt
    }
}

impl ContainersHolder for InnerBareState {
    fn get_containers(
        &self,
    ) -> (
        &IndexMap<String, Surface<Self>>,
        &IndexMap<String, PointCloud<Self>>,
        &IndexMap<String, Segment<Self>>,
    ) {
        (&self.surfaces, &self.clouds, &self.segments)
    }

    fn get_containers_mut(
        &mut self,
    ) -> (
        &mut IndexMap<String, Surface<Self>>,
        &mut IndexMap<String, PointCloud<Self>>,
        &mut IndexMap<String, Segment<Self>>,
        Self::Context<'_>,
        Self::ExtendedContext<'_>,
    ) {
        (
            &mut self.surfaces,
            &mut self.clouds,
            &mut self.segments,
            &mut self.settings,
            (),
        )
    }
}

impl StateTrait for InnerBareState {
    #[cfg(feature = "saves")]
    fn load_from_state_slice(&mut self, data: &[u8]) -> Result<(), ()> {
        let bared: InnerBareStateSerde = serde_cbor::from_slice(data).map_err(|_| ())?;
        self.surfaces = bared.surfaces;
        self.clouds = bared.clouds;
        self.segments = bared.segments;
        self.settings = bared.settings;
        self.camera = bared.camera;
        Ok(())
    }

    #[cfg(feature = "saves")]
    fn save_state_vec(&self) -> Result<Vec<u8>, ()> {
        let surfaces = self.surfaces.clone();
        let clouds = self.clouds.clone();
        let segments = self.segments.clone();
        let bared = InnerBareStateSerde {
            settings: self.settings.clone(),
            camera: self.camera.clone(),
            surfaces,
            clouds,
            segments,
            //ground_level: self.ground.level,
            ground_level: 0.,
        };
        serde_cbor::to_vec(&bared).map_err(|_| ())
    }

    fn get_camera(&self) -> &Camera {
        &self.camera
    }

    fn get_camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    fn get_settings(&self) -> &Settings {
        &self.settings
    }

    fn get_settings_mut(&mut self) -> &mut Settings {
        &mut self.settings
    }
}

pub trait StateTrait:
    GeometryHolder<Surface<Self>, Args = (SurfaceIndices, Vec<[f32; 3]>)>
    + GeometryHolder<PointCloud<Self>, Args = Vec<[f32; 3]>>
    + GeometryHolder<Segment<Self>, Args = (Vec<[f32; 3]>, Vec<[u32; 2]>)>
where
    SurfaceDesc: ShapeDescriptor<Self>,
    PointCloudDesc: ShapeDescriptor<Self>,
    SegmentDesc: ShapeDescriptor<Self>,
{
    #[cfg(feature = "saves")]
    fn load_from_state_slice(&mut self, data: &[u8]) -> Result<(), ()>;

    #[cfg(feature = "saves")]
    fn save_state_vec(&self) -> Result<Vec<u8>, ()>;

    fn get_camera(&self) -> &Camera;

    fn get_camera_mut(&mut self) -> &mut Camera;

    fn get_settings(&self) -> &Settings;

    fn get_settings_mut(&mut self) -> &mut Settings;
}

pub struct State<T>(pub(crate) T);

impl<T: StateTrait> State<T>
where
    SurfaceDesc: ShapeDescriptor<T>,
    PointCloudDesc: ShapeDescriptor<T>,
    SegmentDesc: ShapeDescriptor<T>,
{
    pub(crate) fn new_inner(inner: T) -> Self {
        Self(inner)
    }

    /// Register a new surface. If an existing one with same number of vertices
    /// and same faces exists, previous settings and data are recovered.
    ///
    /// See [`Surface`] and [`SurfaceMut`] for how to add data to the created
    /// shape.
    pub fn register_surface<V: Vertices, I: Into<SurfaceIndices>>(
        &'_ mut self,
        name: impl Into<String>,
        vertices: V,
        indices: I,
    ) -> SurfaceMut<'_, T> {
        self.0
            .register(name.into(), (indices.into(), vertices.into()))
    }

    pub fn get_surface_mut(&'_ mut self, name: &str) -> Option<SurfaceMut<'_, T>> {
        self.0.get_shape_mut(name)
    }

    pub fn get_surface(&self, name: &str) -> Option<&Surface<T>> {
        self.0.get_shape(name)
    }

    pub fn remove_surface(&mut self, name: &str) {
        <T as GeometryHolder<Surface<T>>>::remove_shape(&mut self.0, name);
    }

    /// Register a new point cloud. If an existing one with same number of points
    /// exists, previous settings and data are recovered.
    ///
    /// See [`PointCloud`] and [`PointCloudMut`] for how to add data to the created
    /// shape.
    pub fn register_point_cloud<V: Vertices>(
        &'_ mut self,
        name: impl Into<String>,
        positions: V,
    ) -> PointCloudMut<'_, T> {
        self.0.register(name.into(), positions.into())
    }

    pub fn get_point_cloud_mut(&'_ mut self, name: &str) -> Option<PointCloudMut<'_, T>> {
        self.0.get_shape_mut(name)
    }

    pub fn get_point_cloud(&self, name: &str) -> Option<&PointCloud<T>> {
        self.0.get_shape(name)
    }

    pub fn remove_point_cloud(&mut self, name: &str) {
        <T as GeometryHolder<PointCloud<T>>>::remove_shape(&mut self.0, name);
    }

    /// Register a list of segments. If an existing one with same number of points
    /// and same connextions exists, previous settings and data are recovered.
    ///
    /// See [`Segment`] and [`SegmentMut`] for how to add data to the created
    /// shape.
    ///
    /// Arguments :
    /// * `positions`: segments extremities
    /// * `connections`: segments denoted by extremities indices
    pub fn register_segment<V: Vertices>(
        &'_ mut self,
        name: impl Into<String>,
        positions: V,
        connections: Vec<[u32; 2]>,
    ) -> SegmentMut<'_, T> {
        self.0
            .register(name.into(), (positions.into(), connections))
    }

    pub fn get_segment_mut(&'_ mut self, name: &str) -> Option<SegmentMut<'_, T>> {
        self.0.get_shape_mut(name)
    }

    pub fn get_segment(&self, name: &str) -> Option<&Segment<T>> {
        self.0.get_shape(name)
    }

    pub fn remove_segment(&mut self, name: &str) {
        <T as GeometryHolder<Segment<T>>>::remove_shape(&mut self.0, name);
    }

    /// Load app state from given file content.
    #[cfg_attr(docsrs, doc(cfg(all(feature = "saves", not(target_arch = "wasm32")))))]
    #[cfg(all(feature = "saves", not(target_arch = "wasm32")))]
    pub fn load_from_state_file(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), ()> {
        let data = std::fs::read(path.as_ref()).map_err(|_| ())?;
        self.0.load_from_state_slice(&data)
    }

    /// Load app state from given buffer.
    #[cfg_attr(docsrs, doc(cfg(feature = "saves")))]
    #[cfg(feature = "saves")]
    pub fn load_from_state_slice(&mut self, data: &[u8]) -> Result<(), ()> {
        self.0.load_from_state_slice(data)
    }

    /// Save current state in cbor into chosen file.
    #[cfg_attr(docsrs, doc(cfg(all(feature = "saves", not(target_arch = "wasm32")))))]
    #[cfg(all(feature = "saves", not(target_arch = "wasm32")))]
    pub fn save_state_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), ()> {
        let data = self.0.save_state_vec()?;
        std::fs::write(path.as_ref(), &data).map_err(|_| ())
    }

    /// Save current state in cbor, downloaded in browser.
    #[cfg_attr(docsrs, doc(cfg(all(feature = "saves", target_arch = "wasm32"))))]
    #[cfg(all(feature = "saves", target_arch = "wasm32"))]
    pub fn save_state(&self) -> Result<(), ()> {
        let data = self.0.save_state_vec()?;
        save_state("deuxfleurs.cbor", &data);
        Ok(())
    }

    /// Save current state in cbor into buffer.
    #[cfg_attr(docsrs, doc(cfg(feature = "saves")))]
    #[cfg(feature = "saves")]
    pub fn save_state_vec(&self) -> Result<Vec<u8>, ()> {
        self.0.save_state_vec()
    }

    pub fn get_settings_mut(&mut self) -> &mut Settings {
        self.0.get_settings_mut()
    }

    pub fn get_settings(&self) -> &Settings {
        self.0.get_settings()
    }

    pub fn set_camera(&mut self, eye: [f32; 3], target: [f32; 3], up: [f32; 3]) {
        self.0
            .get_camera_mut()
            .set_from_eye_target_up(eye, target, up);
    }

    /// Result is `(eye, target, up)`.
    pub fn get_camera(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        self.0.get_camera().as_eye_target_up()
    }
}

/// Starting point to build the app.
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
    /// * `id`: serves as window title, or id shape to attach to. If `None` uses `"State"`.
    ///
    /// ```
    /// use deuxfleurs::load_mesh;
    ///
    /// # fn main() {
    /// #     pollster::block_on(run());
    /// # }
    /// # pub async fn run() {
    /// let (spot_v, spot_f) = load_mesh("examples/assets/spot.obj").await.unwrap();
    /// let mut handle = deuxfleurs::init();
    /// handle.register_surface("Spot", spot_v, spot_f);
    /// let mut handle = handle.run(1920, 1080, Some("deuxfleurs"));
    /// # }
    /// ```
    pub fn run<S: Into<String>>(self, width: u32, height: u32, id: Option<S>) {
        self.run_with_callback(width, height, id, |_, _| ());
    }

    /// Run the app without a window. Allows running the app in environment where no
    /// display is available and taking screenshots automatically.
    ///
    /// Currently only available on non wasm targets, as webGL requires a context.
    ///
    /// ```
    /// use deuxfleurs::load_mesh;
    ///
    /// # fn main() {
    /// #     pollster::block_on(run());
    /// # }
    /// # pub async fn run() {
    /// let (spot_v, spot_f) = load_mesh("examples/assets/spot.obj").await.unwrap();
    /// let mut handle = deuxfleurs::init();
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
    pub fn run_with_callback<S: Into<String>, U: FnMut(&mut egui::Ui, &mut RunningState)>(
        self,
        width: u32,
        height: u32,
        id: Option<S>,
        callback: U,
    ) {
        StateWrapper::run(self, width, height, id.map(Into::into), callback);
    }
}

/// Holds the application state. Starting point to add visualization datas.
pub type RunningState = State<InnerGraphicalState>;

struct StateWrapper<T: FnMut(&mut egui::Ui, &mut RunningState)> {
    init_state: Option<InitialState>,
    state: Option<RunningState>,
    ui: Option<crate::ui::UI>,
    clipboard: Option<Clipboard>,
    callback: T,
    id: String,
    width: u32,
    height: u32,
    proxy: EventLoopProxy<UserEvent>,
}

pub(crate) enum UserEvent {
    #[cfg(feature = "surface_button")]
    LoadMesh(Vec<[f32; 3]>, crate::types::SurfaceIndices, String),
    #[cfg(feature = "saves")]
    LoadState(InnerBareStateSerde),
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

    /// Take a screenshot of the scene. The screenshot is then
    /// saved on disk under the name `screenshot_[nnn].png`, with
    /// `nnn` incrementing each time.
    pub fn screenshot(&mut self) {
        self.0.screenshot();
    }

    /// Take a screenshot of the scene. The screenshot is then
    /// returned as a vector oy bytes, each storing `r` `g` `b`
    /// `a` (in order) values of each pixel.
    ///
    /// Should not fail, unless internal buffer storage is
    /// messed up.
    pub fn screenshot_to_buffer(&mut self) -> Result<Vec<u8>, ()> {
        self.0.screenshot_to_buffer()
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
    fn run(init_state: InitialState, width: u32, height: u32, id: Option<String>, callback: T) {
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
            callback,
            ui: None,
            id,
            width,
            height,
            proxy,
        };
        event_loop.run_app(&mut app).unwrap();
    }
}
