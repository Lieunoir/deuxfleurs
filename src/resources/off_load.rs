use std::{
    fmt,
    fs::File,
    io::{BufReader, prelude::*},
    path::Path,
    str::FromStr,
};

use crate::types::SurfaceIndices;

type Float = f32;

unsafe fn parse_float3(slice: &[u8]) -> (usize, [Float; 3]) {
    unsafe {
        let mut start = 0;
        while slice[start] == b' ' {
            start += 1;
        }
        let mut sep = find_white_space(&slice[start..]).unwrap();
        let f1 =
            FromStr::from_str(std::str::from_utf8_unchecked(&slice[start..(start + sep)])).unwrap();
        start = start + sep + 1;
        while slice[start] == b' ' {
            start += 1;
        }
        sep = find_white_space(&slice[start..]).unwrap();
        let f2 =
            FromStr::from_str(std::str::from_utf8_unchecked(&slice[start..(start + sep)])).unwrap();
        start = start + sep + 1;
        while slice[start] == b' ' {
            start += 1;
        }
        sep = find_white_space(&slice[start..]).unwrap();
        let f3 =
            FromStr::from_str(std::str::from_utf8_unchecked(&slice[start..(start + sep)])).unwrap();
        let mut off = start + sep;
        while slice[off] == b' ' || slice[off] == b'\r' {
            off += 1;
        }

        let arr: [Float; 3] = [f1, f2, f3];

        (off, arr)
    }
}

fn find_newline(slice: &[u8]) -> Option<usize> {
    for (i, &v) in slice.iter().enumerate() {
        if v == b'\n' {
            return Some(i);
        }
    }
    None
}

fn find_white_space(slice: &[u8]) -> Option<usize> {
    for (i, &v) in slice.iter().enumerate() {
        if v == b' ' || v == b'\n' || v == b'\r' {
            return Some(i);
        }
    }
    None
}

fn get_line_start(slice: &[u8]) -> Option<usize> {
    for (i, char) in slice.iter().enumerate() {
        if *char != b' ' {
            if *char == b'#' {
                return None;
            } else if *char == b'\n' || *char == b'\r' {
                return None;
            } else {
                return Some(i);
            }
        }
    }
    None
}

fn parse_int(data: &[u8], pos_sz: u32) -> Option<(u32, usize)> {
    if data.len() > 0 {
        if data[0] == b'-' {
            let mut acc = 0;
            let mut i = 1;
            for &value in &data[i..] {
                if value < b'0' || value > b'9' {
                    break;
                }
                i += 1;
                acc = acc * 10 + (value - b'0') as u32;
            }
            Some((pos_sz - acc, i))
        } else {
            if data[0] == b'+' {
                let mut acc = 0;
                let mut i = 1;
                for &value in &data[i..] {
                    if value < b'0' || value > b'9' {
                        break;
                    }
                    i += 1;
                    acc = acc * 10 + (value - b'0') as u32;
                }
                Some((acc, i))
            } else {
                let mut acc = 0;
                let mut i = 0;
                for &value in &data[i..] {
                    if value < b'0' || value > b'9' {
                        break;
                    }
                    i += 1;
                    acc = acc * 10 + (value - b'0') as u32;
                }
                Some((acc, i))
            }
        }
    } else {
        None
    }
}

fn parse_face_pos(
    face_str: &[u8],
    mode: &mut FaceMode,
    indices: &mut Vec<u32>,
    strides: &mut Vec<u8>,
    nf: usize,
) -> usize {
    let mut off = 0;
    let mut data = face_str;
    while data.len() > 0 && data[0] == b' ' {
        data = &data[1..];
    }

    let (face_len, mut endword) = parse_int(data, 0).unwrap();
    if face_len >= 3 && *mode != FaceMode::Polygon {
        if *mode == FaceMode::Undetermined {
            if face_len == 3 {
                *mode = FaceMode::Triangle;
            } else if face_len == 4 {
                *mode = FaceMode::Quad;
            } else {
                *mode = FaceMode::Polygon;
            }
        } else if *mode == FaceMode::Triangle && face_len != 3 {
            //add missing strides
            *strides = vec![3; (indices.len() - face_len as usize) / 3];
            strides.reserve(3 * nf - strides.len());
            *mode = FaceMode::Polygon;
        } else if *mode == FaceMode::Quad && face_len != 4 {
            //add missing strides
            *strides = vec![4; (indices.len() - face_len as usize) / 4];
            *mode = FaceMode::Polygon;
            strides.reserve(2 * nf - strides.len());
        }
    }
    if face_len >= 3 && *mode == FaceMode::Polygon {
        strides.push(face_len as u8);
    }

    while endword < data.len() && data[endword] == b' ' {
        endword += 1;
    }
    off += endword;
    data = &data[endword..];

    for _ in 0..face_len {
        let (v, mut endword) = parse_int(data, 0).unwrap();
        indices.push(v);
        while endword < data.len() && data[endword] == b' ' {
            endword += 1;
        }
        off += endword;
        data = &data[endword..];
    }
    off
}

fn parse_header(buf: &[u8]) -> (usize, usize, usize) {
    let (nv, mut endword) = parse_int(buf, 0).unwrap();
    while endword < buf.len() && buf[endword] == b' ' {
        endword += 1;
    }
    let (nf, endword) = parse_int(&buf[endword..], 0).unwrap();
    (nv as usize, nf as usize, endword)
}

pub fn load_off(file_name: impl AsRef<Path>) -> (Vec<[Float; 3]>, SurfaceIndices) {
    let file = match File::open(file_name.as_ref()) {
        Ok(f) => f,
        Err(_e) => {
            panic!()
            //return Err(LoadError::OpenFileFailed);
        }
    };
    let mut reader = BufReader::new(file);
    load_off_buf(&mut reader)
}

pub fn load_off_buf<B>(reader: &mut B) -> (Vec<[Float; 3]>, SurfaceIndices)
where
    B: BufRead,
{
    let mut line_number = 0;
    let mut nv = 0;
    let mut nf = 0;
    let mut vertices = Vec::new();
    let mut mode = FaceMode::Undetermined;
    let mut indices: Vec<u32> = Vec::new();
    let mut strides: Vec<u8> = Vec::new();
    const BUFFER_SIZE: usize = 65536;
    let mut buf = [0; BUFFER_SIZE];
    let mut start = 0;
    'outer: while let Ok(size) = reader.read(&mut buf[start..]) {
        if size == 0 && start == 0 {
            break;
        }
        let end = start + size;
        let mut last = end - 1;
        while buf[last] != b'\n' && last > 0 {
            last -= 1;
        }
        if buf[last] != b'\n' {
            break;
        }
        last += 1;

        let mut i = 0;
        while i < last {
            if let Some(line_start) = get_line_start(&buf[i..]) {
                i += line_start;
                if line_number == 0 {
                    if &buf[i..i + 3] == &[b'O', b'F', b'F'] {
                        i += find_newline(&buf[3..]).unwrap() + 4;
                        continue;
                    } else {
                        let endword;
                        (nv, nf, endword) = parse_header(&buf[i..]);
                        vertices.reserve(nv);
                        indices.reserve(3 * nf);
                        line_number += 1;
                        i += find_newline(&buf[i + endword..]).unwrap() + endword + 1;
                    }
                } else if line_number < nv + 1 {
                    let (off, pos) = unsafe { parse_float3(&buf[i..]) };
                    line_number += 1;
                    vertices.push(pos);
                    i += off + 1;
                } else if line_number < 1 + nv + nf {
                    let off = parse_face_pos(&buf[i..], &mut mode, &mut indices, &mut strides, nf);
                    line_number += 1;
                    i += 1 + off;
                } else {
                    break 'outer;
                }
            } else {
                i += find_newline(&buf[1..]).unwrap() + 1;
            }
        }

        start = end - last;
        buf.copy_within(last..end, 0);
    }

    let indices = if mode == FaceMode::Polygon {
        (indices, strides).into()
    } else if mode == FaceMode::Quad {
        indices
            .chunks(4)
            .map(|face| face.try_into().unwrap())
            .collect::<Vec<[u32; 4]>>()
            .into()
    } else {
        indices
            .chunks(3)
            .map(|face| face.try_into().unwrap())
            .collect::<Vec<[u32; 3]>>()
            .into()
    };
    (vertices, indices)
}

#[derive(PartialEq)]
enum FaceMode {
    Triangle,
    Quad,
    Polygon,
    Undetermined,
}
