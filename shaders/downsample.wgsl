struct DownsampleParams {
    supersampling: f32,
    highResWidth: f32,
    highResHeight: f32,
};

@group(0) @binding(0) var<uniform> params: DownsampleParams;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var inputSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn downsampleVertex(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );
    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = pos[vertexIndex] * 0.5 + 0.5;
    output.uv.y = 1.0 - output.uv.y;
    return output;
}

@fragment
fn downsampleFragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Proper box filter: sample a grid of pixels and average them
    // Calculate texel size in high-res texture
    let ss = params.supersampling;
    let texelSizeX = 1.0 / params.highResWidth;
    let texelSizeY = 1.0 / params.highResHeight;
    
    // Convert UV to pixel coordinates in high-res texture
    let pixelX = uv.x * params.highResWidth;
    let pixelY = uv.y * params.highResHeight;
    
    // Find the top-left corner of the box we're averaging
    let boxStartX = floor(pixelX / ss) * ss;
    let boxStartY = floor(pixelY / ss) * ss;
    
    // Sample grid of pixels and average (box filter)
    // For 2x supersampling, sample 2x2 grid. For 4x, sample 4x4, etc.
    var colorSum = vec4<f32>(0.0);
    let sampleCount = ss * ss;
    
    // Sample a grid of pixels at exact texel centers
    for (var y = 0.0; y < ss; y += 1.0) {
        for (var x = 0.0; x < ss; x += 1.0) {
            // Calculate exact texel center position
            let samplePixelX = boxStartX + x + 0.5;
            let samplePixelY = boxStartY + y + 0.5;
            
            // Convert back to UV coordinates
            let sampleUV = vec2<f32>(
                samplePixelX / params.highResWidth,
                samplePixelY / params.highResHeight
            );
            
            colorSum += textureSample(inputTexture, inputSampler, sampleUV);
        }
    }
    
    // Average the samples
    return colorSum / f32(sampleCount);
}
