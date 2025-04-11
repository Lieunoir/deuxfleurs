use std::{
    fmt,
    fs::File,
    io::{prelude::*, BufReader},
    path::Path,
    str::FromStr,
};

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
                Some((acc - 1, i))
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
                Some((acc - 1, i))
            }
        }
    } else {
        None
    }
}

fn parse_face_pos(
    //face_str: SplitAsciiWhitespace,
    face_str: &[u8],
    mode: &mut FaceMode,
    indices: &mut Vec<u32>,
    _tex_indices: &mut Vec<u32>,
    _n_indices: &mut Vec<u32>,
    strides: &mut Vec<u8>,
    pos_sz: u32,
    _tex_sz: u32,
    _norm_sz: u32,
) -> usize {
    let mut i = 0;
    let mut data = face_str;

    let mut off = 0;
    while data.len() > 0 && data[0] == b' ' {
        data = &data[1..];
    }
    while let Some((v_i, mut endword)) = parse_int(data, pos_sz) {
        indices.push(v_i);
        i += 1;
        if endword == data.len() {
            break;
        }
        if data[endword] == b'/' {
            match find_white_space(&data[(endword + 1)..]) {
                Some(value) => endword += 1 + value,
                None => break,
            }
        }

        while endword < data.len() && data[endword] == b' ' {
            endword += 1;
        }
        off += endword;
        if data[endword] == b'\r' || data[endword] == b'\n' {
            break;
        }
        data = &data[endword..];
    }
    //if data.len() > 0 {
    //    let v_i = VertexIndices::parse_pos(data, pos_sz).unwrap();
    //    indices.push(v_i);
    //    i += 1;
    //}
    if i >= 3 && *mode != FaceMode::Polygon {
        if *mode == FaceMode::Undetermined {
            if i == 3 {
                *mode = FaceMode::Triangle;
            } else if i == 4 {
                *mode = FaceMode::Quad;
            } else {
                *mode = FaceMode::Polygon;
            }
        } else if *mode == FaceMode::Triangle && i != 3 {
            //add missing strides
            *strides = vec![3; (indices.len() - i) / 3];
            strides.reserve(2 * pos_sz as usize - strides.len());
            *mode = FaceMode::Polygon;
        } else if *mode == FaceMode::Quad && i != 4 {
            //add missing strides
            *strides = vec![4; (indices.len() - i) / 4];
            *mode = FaceMode::Polygon;
            strides.reserve(2 * pos_sz as usize - strides.len());
        }
    }
    if i >= 3 && *mode == FaceMode::Polygon {
        strides.push(i as u8);
    }
    off
}

pub fn load_obj<P>(file_name: P) -> (Vec<[Float; 3]>, Vec<u32>, Vec<u8>)
where
    P: AsRef<Path> + fmt::Debug,
{
    let file = match File::open(file_name.as_ref()) {
        Ok(f) => f,
        Err(_e) => {
            panic!()
            //return Err(LoadError::OpenFileFailed);
        }
    };
    let mut reader = BufReader::new(file);
    load_obj_buf(&mut reader)
}

pub fn load_obj_buf<B>(reader: &mut B) -> (Vec<[Float; 3]>, Vec<u32>, Vec<u8>)
where
    B: BufRead,
{
    let arch = pulp::Arch::new();
    arch.dispatch(|| {
        let mut tmp_pos = Vec::new();
        let mut mode = FaceMode::Undetermined;
        let mut indices: Vec<u32> = Vec::new();
        let mut tex_indices: Vec<u32> = Vec::new();
        let mut n_indices: Vec<u32> = Vec::new();
        let mut strides: Vec<u8> = Vec::new();
        const BUFFER_SIZE: usize = 65536;
        let mut buf = [0; BUFFER_SIZE];
        let mut encountered_f = false;
        let mut start = 0;
        while let Ok(size) = reader.read(&mut buf[start..]) {
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
                match buf[i] {
                    b'v' => match buf[i + 1] {
                        b' ' => {
                            let (off, pos) = unsafe { parse_float3(&buf[i + 2..]) };
                            tmp_pos.push(pos);
                            i += off + 2;
                        }
                        _ => i += find_newline(&buf[i + 1..]).unwrap() + 2,
                    },
                    b'f' => {
                        if !encountered_f {
                            encountered_f = true;
                            indices.reserve(tmp_pos.len() * 2);
                        }
                        let off = parse_face_pos(
                            &buf[i + 2..],
                            &mut mode,
                            &mut indices,
                            &mut tex_indices,
                            &mut n_indices,
                            &mut strides,
                            tmp_pos.len() as u32,
                            0,
                            0,
                        );
                        i += 2 + off;
                    }
                    _ => i += find_newline(&buf[i..]).unwrap() + 1,
                }
            }

            start = end - last;
            buf.copy_within(last..end, 0);
        }
        (tmp_pos, indices, strides)
    })
}

#[derive(PartialEq)]
enum FaceMode {
    Triangle,
    Quad,
    Polygon,
    Undetermined,
}
