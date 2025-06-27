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
