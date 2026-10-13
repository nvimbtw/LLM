struct Parameters {
    transform: vec4<f32>,
    motion: vec4<f32>,
    palette: array<vec4<f32>, 4>,
    nodes: array<vec4<f32>, 31>,
}
@group(0) @binding(0) var<uniform> art: Parameters;

struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) point: vec2<f32>,
}

@vertex fn vertex(@builtin(vertex_index) index: u32) -> Vertex {
    let vertices = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: Vertex;
    out.position = vec4(vertices[index], 0.0, 1.0);
    out.point = vertices[index];
    return out;
}

fn combine(node: vec4<f32>, a: f32, b: f32) -> f32 {
    switch u32(node.x) {
        case 0u: { return sin((a + b) * node.y + node.w); }
        case 1u: { return cos((a - b) * node.y + node.w); }
        case 2u: { return a * b; }
        default: { return mix(a, b, node.z); }
    }
}
// Constant node references let the GPU keep intermediates in registers instead
// of dynamically indexing a per-pixel array. The depth-four grammar is unchanged.
fn expression(point: vec2<f32>) -> vec3<f32> {
    let n30 = art.nodes[30];
    let v30 = sin(point.x * n30.y + point.y * n30.z + n30.w);
    let n29 = art.nodes[29];
    let v29 = sin(point.x * n29.y + point.y * n29.z + n29.w);
    let n28 = art.nodes[28];
    let v28 = sin(point.x * n28.y + point.y * n28.z + n28.w);
    let n27 = art.nodes[27];
    let v27 = sin(point.x * n27.y + point.y * n27.z + n27.w);
    let n26 = art.nodes[26];
    let v26 = sin(point.x * n26.y + point.y * n26.z + n26.w);
    let n25 = art.nodes[25];
    let v25 = sin(point.x * n25.y + point.y * n25.z + n25.w);
    let n24 = art.nodes[24];
    let v24 = sin(point.x * n24.y + point.y * n24.z + n24.w);
    let n23 = art.nodes[23];
    let v23 = sin(point.x * n23.y + point.y * n23.z + n23.w);
    let n22 = art.nodes[22];
    let v22 = sin(point.x * n22.y + point.y * n22.z + n22.w);
    let n21 = art.nodes[21];
    let v21 = sin(point.x * n21.y + point.y * n21.z + n21.w);
    let n20 = art.nodes[20];
    let v20 = sin(point.x * n20.y + point.y * n20.z + n20.w);
    let n19 = art.nodes[19];
    let v19 = sin(point.x * n19.y + point.y * n19.z + n19.w);
    let n18 = art.nodes[18];
    let v18 = sin(point.x * n18.y + point.y * n18.z + n18.w);
    let n17 = art.nodes[17];
    let v17 = sin(point.x * n17.y + point.y * n17.z + n17.w);
    let n16 = art.nodes[16];
    let v16 = sin(point.x * n16.y + point.y * n16.z + n16.w);
    let n15 = art.nodes[15];
    let v15 = sin(point.x * n15.y + point.y * n15.z + n15.w);
    let v14 = combine(art.nodes[14], v29, v30);
    let v13 = combine(art.nodes[13], v27, v28);
    let v12 = combine(art.nodes[12], v25, v26);
    let v11 = combine(art.nodes[11], v23, v24);
    let v10 = combine(art.nodes[10], v21, v22);
    let v9 = combine(art.nodes[9], v19, v20);
    let v8 = combine(art.nodes[8], v17, v18);
    let v7 = combine(art.nodes[7], v15, v16);
    let v6 = combine(art.nodes[6], v13, v14);
    let v5 = combine(art.nodes[5], v11, v12);
    let v4 = combine(art.nodes[4], v9, v10);
    let v3 = combine(art.nodes[3], v7, v8);
    let v2 = combine(art.nodes[2], v5, v6);
    let v1 = combine(art.nodes[1], v3, v4);
    let v0 = combine(art.nodes[0], v1, v2);
    return vec3(v0, v1, v2);
}

@fragment fn fragment(input: Vertex) -> @location(0) vec4<f32> {
    let radius = length(input.point);
    let edge = max(fwidth(radius), 0.002);
    let alpha = 1.0 - smoothstep(0.97 - edge, 0.97, radius);
    if (alpha == 0.0) {
        return vec4(0.0);
    }
    let angle = art.transform.w;
    let rotation = mat2x2(cos(angle), sin(angle), -sin(angle), cos(angle));
    var point = rotation * (input.point * 1.8 * art.transform.z) + art.transform.xy;
    if (art.motion.w != 0.0) {
        point.x += sin(point.y * 3.0 + art.motion.z) * art.motion.w;
    }
    if (art.motion.y != 0.0) {
        let distance = point.x - art.motion.x;
        let scan = exp(-distance * distance * 4.0);
        if (art.motion.y < 0.0) {
            let opposite = point.x + art.motion.x;
            let paired = scan - exp(-opposite * opposite * 4.0);
            point.y -= paired * art.motion.y;
        } else {
            point.y += scan * art.motion.y;
        }
    }
    let field = expression(point);
    let ink = 0.5 + field.x * 0.5;
    let strata = 0.5 + 0.5 * sin(field.x * 9.0 + field.y * 3.0);
    let colour = mix(art.palette[0].rgb, art.palette[1].rgb, smoothstep(0.12, 0.84, ink));
    let veined = mix(colour, art.palette[2].rgb, smoothstep(0.70, 0.96, strata) * 0.65);
    let lit = mix(veined, art.palette[3].rgb, smoothstep(0.65, 1.0, field.z) * 0.55);
    let depth = 0.86 + 0.14 * sqrt(max(0.0, 1.0 - radius * radius));
    return vec4(lit * depth * alpha, alpha);
}
