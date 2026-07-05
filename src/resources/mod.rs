use crate::types::SurfaceIndices;
use std::{
    io::{BufReader, Cursor},
    path::Path,
};

mod obj_load;
mod off_load;

#[cfg(target_arch = "wasm32")]
async fn fetch(file_name: &Path) -> Option<String> {
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};
    let window = web_sys::window().unwrap();
    let location = window.location().origin().ok()?;
    let url = format!("{location}/{}", file_name.to_string_lossy());
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

async fn load_string(file_name: &Path) -> Option<String> {
    cfg_select! {
        target_arch = "wasm32" =>  fetch(file_name).await,
        _ => {
            let text = std::fs::read_to_string(file_name).ok()?;
            Some(text)
        }
    }
}

/// Helper to load a mesh from an obj or off file
pub async fn load_mesh(file_name: impl AsRef<Path>) -> Option<(Vec<[f32; 3]>, SurfaceIndices)> {
    let file_text = load_string(file_name.as_ref()).await?;
    let obj_cursor = Cursor::new(file_text);
    let mut file_reader = BufReader::new(obj_cursor);
    match file_name.as_ref().extension() {
        Some(ext) => {
            if ext == "obj" {
                Some(obj_load::load_obj_buf(&mut file_reader))
            } else if ext == "off" {
                Some(off_load::load_off_buf(&mut file_reader))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Helper to load a mesh from an obj or off file
pub fn load_mesh_blocking(file_name: impl AsRef<Path>) -> Option<(Vec<[f32; 3]>, SurfaceIndices)> {
    match file_name.as_ref().extension() {
        Some(ext) => {
            if ext == "obj" {
                Some(obj_load::load_obj(file_name))
            } else if ext == "off" {
                Some(off_load::load_off(file_name))
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) async fn parse_preloaded_mesh(
    file_name: impl AsRef<Path>,
    data: Vec<u8>,
) -> Option<(Vec<[f32; 3]>, SurfaceIndices)> {
    let file_cursor = Cursor::new(data);
    let mut file_reader = BufReader::new(file_cursor);
    match file_name.as_ref().extension() {
        Some(ext) => {
            if ext == "obj" {
                Some(obj_load::load_obj_buf(&mut file_reader))
            } else if ext == "off" {
                Some(off_load::load_off_buf(&mut file_reader))
            } else {
                None
            }
        }
        _ => None,
    }
}
