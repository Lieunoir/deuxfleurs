use crate::types::SurfaceIndices;
use std::{
    io::{BufReader, Cursor},
    path::PathBuf,
};

use cfg_if::cfg_if;

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let base = reqwest::Url::parse(&format!(
        "{}/{}/",
        location.origin().unwrap(),
        option_env!("RES_PATH").unwrap_or("assets"),
    ))
    .unwrap();
    base.join(file_name).unwrap()
}

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            log::warn!("Load model on web");

            let url = format_url(file_name);
            let txt = reqwest::get(url)
                .await?
                .text()
                .await?;

            log::warn!("{}", txt);

        } else {
            let path = std::path::Path::new("assets")
                .join(file_name);
            let txt = std::fs::read_to_string(path)?;
        }
    }

    Ok(txt)
}

pub async fn load_mesh(file_name: &str) -> anyhow::Result<(Vec<[f32; 3]>, SurfaceIndices)> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);
    let (v, i, s) = crate::obj_load::load_obj_buf(&mut obj_reader);
    let indices = if s.len() > 0 {
        (i, s).into()
    } else {
        i.chunks(3)
            .map(|face| face.try_into().unwrap())
            .collect::<Vec<[u32; 3]>>()
            .into()
    };

    Ok((v, indices))
}

pub fn load_mesh_blocking(file_name: PathBuf) -> anyhow::Result<(Vec<[f32; 3]>, SurfaceIndices)> {
    let (v, i, s) = crate::obj_load::load_obj(file_name);
    let indices = if s.len() > 0 {
        (i, s).into()
    } else {
        i.chunks(3)
            .map(|face| face.try_into().unwrap())
            .collect::<Vec<[u32; 3]>>()
            .into()
    };

    Ok((v, indices))
}

pub async fn load_preloaded_mesh(data: Vec<u8>) -> anyhow::Result<(Vec<[f32; 3]>, SurfaceIndices)> {
    let obj_cursor = Cursor::new(data);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (v, i, s) = crate::obj_load::load_obj_buf(&mut obj_reader);
    let indices = if s.len() > 0 {
        (i, s).into()
    } else {
        i.chunks(3)
            .map(|face| face.try_into().unwrap())
            .collect::<Vec<[u32; 3]>>()
            .into()
    };

    Ok((v, indices))
}
