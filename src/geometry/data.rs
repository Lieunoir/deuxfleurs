use super::{Context, DataUniformBuilder, GraphicalContext};

pub struct DataMut<'a, T, Ctxt: Context> {
    pub(crate) inner: &'a mut T,
    pub(crate) context: &'a mut Ctxt,
    pub(crate) uniform: Ctxt::DataUniform<'a>,
}

pub type UninitedData<'a, T> = DataMut<'a, T, ()>;
pub type DisplayData<'a, T> = DataMut<'a, T, GraphicalContext<'a>>;

impl<'a, T, Ctxt: Context> DataMut<'a, T, Ctxt> {
    pub(crate) fn convert<U, F: FnOnce(&mut T) -> &mut U>(self, f: F) -> DataMut<'a, U, Ctxt> {
        DataMut {
            inner: f(self.inner),
            uniform: self.uniform,
            context: self.context,
        }
    }
}

pub trait DataMutTrait {
    fn update_data_settings(&mut self);
}

impl<'a, T> DataMutTrait for UninitedData<'a, T> {
    fn update_data_settings(&mut self) {}
}

impl<'a, T> DataMutTrait for DisplayData<'a, T>
where
    T: DataUniformBuilder,
{
    fn update_data_settings(&mut self) {
        self.uniform
            .as_ref()
            .map(|uniform| self.inner.refresh_buffer(self.context.queue, uniform));
    }
}

pub trait NewAttachedGeometry {
    type UpgradedAttachedGeometry;

    fn init(
        self,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self::UpgradedAttachedGeometry;
}

impl<Ctxt: Context> AttachedGeometry<Ctxt> for () {
    type Args = ();
    type Settings = ();

    fn new(
        _name: String,
        _args: Self::Args,
        _context: &mut Ctxt,
        _transform_layout: &Ctxt::TransformLayout,
    ) -> Self {
        ()
    }

    fn get_settings(&mut self) -> &mut Self::Settings {
        self
    }
}

pub struct EmptyAttached(());

impl NewAttachedGeometry for () {
    type UpgradedAttachedGeometry = EmptyAttached;

    fn init(
        self,
        _device: &wgpu::Device,
        _camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        _transform_bind_group_layout: &wgpu::BindGroupLayout,
        _color_format: wgpu::TextureFormat,
    ) -> Self::UpgradedAttachedGeometry {
        EmptyAttached(())
    }
}

impl<Ctxt: Context> AttachedGeometry<Ctxt> for EmptyAttached {
    type Args = ();
    type Settings = ();

    fn new(
        _name: String,
        _args: Self::Args,
        _context: &mut Ctxt,
        _transform_layout: &Ctxt::TransformLayout,
    ) -> Self {
        EmptyAttached(())
    }

    fn get_settings(&mut self) -> &mut Self::Settings {
        &mut self.0
    }
}

pub trait NamedSettings: Default + DataUniformBuilder {
    fn set_name(self, name: &str) -> Self;

    fn draw_ui(&mut self, ui: &mut egui::Ui, rebuild_pipeline: &mut bool) -> bool;
}

pub trait ElementGeometry {
    type Args;

    fn new(args: Self::Args) -> Self;

    fn can_be_replaced_by(&self, _other: &Self) -> bool {
        false
    }

    fn get_positions(&self) -> &[[f32; 3]];

    fn get_total_elements(&self) -> u32;
}

pub trait AttachedGeometry<Ctxt: Context> {
    type Args;
    type Settings;

    fn new(
        name: String,
        args: Self::Args,
        context: &mut Ctxt,
        transform_layout: &Ctxt::TransformLayout,
    ) -> Self;

    fn shown(&self) -> bool {
        false
    }

    fn show(&mut self, _show: bool, _refresh_screen: &mut bool) {}

    fn draw_ui(
        &mut self,
        _ui: &mut egui::Ui,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        _color_format: wgpu::TextureFormat,
        _refresh_screen: &mut bool,
    ) {
    }

    fn render<'c, 'd>(&'c self, _render_pass: &mut wgpu::RenderPass<'d>)
    where
        'c: 'd,
    {
    }

    fn get_settings(&mut self) -> &mut Self::Settings;
}
