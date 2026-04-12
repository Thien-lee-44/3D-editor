#version 330 core

// Memory Layout Specifications
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aColor; 

// Output streams representing interpolated fragments
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;
out vec3 VertColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// Normal matrix required to invert non-uniform spatial scaling distortions
uniform mat3 normalMatrix; 

void main() {
    // Project geometry into absolute World-Space for accurate lighting calculations
    FragPos = vec3(model * vec4(aPos, 1.0));
    
    // Resolve final Vertex Position within the rasterizer's clip space
    gl_Position = projection * view * vec4(FragPos, 1.0);
    
    // Compute World-Space surface normals strictly via normal matrix
    Normal = normalMatrix * aNormal; 
    TexCoords = aTexCoords;
    
    // Vertex Color sanitation logic (Defaults to white if unbound)
    if (aColor.r < 0.0) {
        VertColor = vec3(1.0, 1.0, 1.0);
    } else {
        VertColor = aColor;
    }
}