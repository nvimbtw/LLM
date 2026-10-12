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

// A fixed tree keeps its grammar while tokens change the continuous parameters.
// Interpolating topology would introduce discrete jumps in an unfinished morph.
fn expression(point: vec2<f32>) -> vec3<f32> {
    var values: array<f32, 31>;
    for (var step = 0u; step < 31u; step++) {
        let i = 30u - step;
        let node = art.nodes[i];
        if (i >= 15u) {
            values[i] = sin(point.x * node.y + point.y * node.z + node.w);
        } else {
            let a = values[i * 2u + 1u];
            let b = values[i * 2u + 2u];
            switch u32(node.x) {
                case 0u: { values[i] = sin((a + b) * node.y + node.w); }
                case 1u: { values[i] = cos((a - b) * node.y + node.w); }
                case 2u: { values[i] = a * b; }
                default: { values[i] = mix(a, b, node.z); }
            }
        }
    }
    return vec3(values[0], values[1], values[2]);
}

@fragment fn fragment(input: Vertex) -> @location(0) vec4<f32> {
    let radius = length(input.point);
    let edge = max(fwidth(radius), 0.002);
    let alpha = 1.0 - smoothstep(0.97 - edge, 0.97, radius);
    let angle = art.transform.w;
    let rotation = mat2x2(cos(angle), sin(angle), -sin(angle), cos(angle));
    var point = rotation * (input.point * 1.8 * art.transform.z) + art.transform.xy;
    point.x += sin(point.y * 3.0 + art.motion.z) * art.motion.w;
    let scan = exp(-pow(point.x - art.motion.x, 2.0) * 4.0);
    let paired = scan - exp(-pow(point.x + art.motion.x, 2.0) * 4.0);
    point.y += select(scan, -paired, art.motion.y < 0.0) * art.motion.y;
    let field = expression(point);
    let ink = 0.5 + field.x * 0.5;
    let strata = 0.5 + 0.5 * sin(field.x * 9.0 + field.y * 3.0);
    let colour = mix(art.palette[0].rgb, art.palette[1].rgb, smoothstep(0.12, 0.84, ink));
    let veined = mix(colour, art.palette[2].rgb, smoothstep(0.70, 0.96, strata) * 0.65);
    let lit = mix(veined, art.palette[3].rgb, smoothstep(0.65, 1.0, field.z) * 0.55);
    let depth = 0.86 + 0.14 * sqrt(max(0.0, 1.0 - radius * radius));
    return vec4(lit * depth * alpha, alpha);
}
