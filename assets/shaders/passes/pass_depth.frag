#version 330 core
out vec4 FragColor;

uniform int isOrthographic;
uniform float near;
uniform float far;

void main() {
    float depth = gl_FragCoord.z;
    if (isOrthographic == 0) {
        float ndc = depth * 2.0 - 1.0;
        depth = (2.0 * near * far) / (far + near - ndc * (far - near));
        depth = depth / far; 
    }
    FragColor = vec4(vec3(depth), 1.0);
}