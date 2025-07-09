use crate::types::SurfaceIndices;
use cfg_if::cfg_if;
use std::{
    io::{BufReader, Cursor},
    path::PathBuf,
};

#[cfg(target_arch = "wasm32")]
async fn fetch(file_name: &str) -> Option<String> {
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};
    let window = web_sys::window().unwrap();
    let location = window.location().origin().ok()?;
    let url = format!("{location}/{file_name}");
    let opts = RequestInit::new();
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(&url, &opts).ok()?;

    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .ok()?;

    // `resp_value` is a `Response` object.
    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into().unwrap();

    // Convert this other `Promise` into a rust `Future`.
    let json = JsFuture::from(resp.text().ok()?).await.ok()?;
    json.as_string()
}
async fn load_string(file_name: &str) -> Option<String> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            fetch(file_name).await

        } else {
            let path = std::path::Path::new(file_name);
            let text = std::fs::read_to_string(path).ok()?;
            Some(text)
        }
    }
}

/// Helper to load a mesh from an obj file
pub async fn load_mesh(file_name: &str) -> Option<(Vec<[f32; 3]>, SurfaceIndices)> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);
    Some(crate::obj_load::load_obj_buf(&mut obj_reader))
}

/// Helper to load a mesh from an obj file
pub fn load_mesh_blocking(file_name: PathBuf) -> Option<(Vec<[f32; 3]>, SurfaceIndices)> {
    Some(crate::obj_load::load_obj(file_name))
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn parse_preloaded_mesh(data: Vec<u8>) -> Option<(Vec<[f32; 3]>, SurfaceIndices)> {
    let obj_cursor = Cursor::new(data);
    let mut obj_reader = BufReader::new(obj_cursor);

    Some(crate::obj_load::load_obj_buf(&mut obj_reader))
}
